import torch
import math
import contextlib
from torch.distributed import ReduceOp


class SASSHA(torch.optim.Optimizer):
    """Implements the Sharpness-Aware Second-Order optimization with Stable Hessian Approximation (SASSHA) algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        hessian_power_scheduler (None): Update the Hessian power at every training step. Initially, set it to None, and later you can replace it.
        lr (float, optional): Learning rate.
        betas (Tuple[float, float], optional): Coefficients for computing moving averages of gradient and Hessian.
        weight_decay (float, optional): Weight decay (L2 penalty).
        rho (float, optional): Size of the neighborhood for computing the max loss
        hessian_power (float, optional): Exponent of the Hessian in the update rule.
        lazy_hessian (int, optional): Number of optimization steps to perform before updating the Hessian.
        n_samples (int, optional): Number of samples to draw for the Hutchinson approximation.
        perturb_eps (float, optional): Small value for perturbations in Hessian trace computation.
        eps (float, optional): Term added to the denominator to improve numerical stability.
        adaptive (bool, optional): set this argument to True if you want to use an experimental implementation of element-wise Adaptive SAM. Default is False.
        grad_reduce (str, optional): Reduction method for gradients ('mean' or 'sum'). Default is 'mean'.
        seed (int, optional): Random seed for reproducibility. Default is 0.
        **kwargs: Additional keyword arguments for compatibility with other optimizers.
    """

    def __init__(self, params, 
                hessian_power_scheduler=None,
                lr=0.15,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                rho=0.0,
                lazy_hessian=10,
                n_samples=1,
                perturb_eps=1e-12,
                eps=1e-4,
                adaptive=False,
                grad_reduce='mean',
                seed=0,
                **kwargs):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.hessian_power_scheduler = hessian_power_scheduler
        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.adaptive = adaptive
        self.seed = seed

        defaults = dict(lr=lr,
                        betas=betas,
                        weight_decay=weight_decay,
                        rho=rho,
                        perturb_eps=perturb_eps,
                        eps=eps)

        super(SASSHA, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
        
        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(self.seed)


    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.lazy_hessian == 0:
                p.hess.zero_()


    @torch.no_grad()
    def update_hessian_power(self):
        """
        Update the Hessian power at every training step.
        """
        if self.hessian_power_scheduler is not None:
            self.hessian_power_t = self.hessian_power_scheduler.step()
        else:
            self.hessian_power_t = None
        return self.hessian_power_t


    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:  # compute a new Hessian once per 'lazy hessian' steps
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)

        grads = [p.grad for p in params]

        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            # Rademacher distribution {-1.0, 1.0}
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < last_sample)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples
                
    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["perturb_eps"])

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])
    
    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm
    
    @torch.no_grad()
    def _sync_gradients(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if torch.distributed.is_initialized(): # synchronize final gardients
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return
    
    @torch.no_grad()
    def _sync_hessians(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.hess is None: continue
                if torch.distributed.is_initialized(): # synchronize final hessian
                    if not p.hess.is_contiguous():
                        p.hess = p.hess.contiguous()
                    
                    if self.manual_average:
                        torch.distributed.all_reduce(p.hess, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.hess.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.hess, op=self.grad_reduce)
        return
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        self.update_hessian_power()
        
        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()
        self._sync_gradients()
        self._sync_hessians()

        # prepare to update parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue
                
                p.hess = p.hess.abs().clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]
                # State initialization
                if len(state) == 2:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                    state['bias_correction2'] = 0
                                          
                exp_avg, exp_hessian_diag = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']

                if (state['hessian step']-1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bias_correction2 ** self.hessian_power_t
    
                step_size = group['lr'] / bias_correction1
                step_size_neg = -step_size

                denom = ((exp_hessian_diag**self.hessian_power_t) / state['bias_correction2']).add_(group['eps'])

                # make update
                p.addcdiv_(exp_avg, denom, value=step_size_neg)

        return loss
