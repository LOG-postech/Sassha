import torch
import math
import contextlib
from torch.distributed import ReduceOp


class SASSHA(torch.optim.Optimizer):
    """Implements the Sharpness-Aware Second-Order optimization with Stable Hessian Approximation (SASSHA) algorithm.
    """

    def __init__(self, params, 
                hessian_power_scheduler=None,
                lr=0.15,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                rho_scheduler=None,
                lazy_hessian=10,
                n_samples=1,
                perturb_eps=1e-12,
                eps=1e-4,
                adaptive=False,
                grad_reduce='sum',
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

        self.hessian_power_scheduler = hessian_power_scheduler  # 이후 외부에서 교체 가능
        self.rho_scheduler = rho_scheduler
        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.adaptive = adaptive
        self.seed = seed

        defaults = dict(lr=lr,
                        betas=betas,
                        weight_decay=weight_decay,
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
            if not isinstance(p.hess, float):
                p.hess.zero_()


    @torch.no_grad()
    def update_hessian_power(self):
        if self.hessian_power_scheduler is not None:
            self.hessian_power_t = self.hessian_power_scheduler.step()
        else:
            self.hessian_power_t = None
        return self.hessian_power_t
    
    
    @torch.no_grad()
    def update_rho_t(self):
        if self.rho_scheduler is not None:
            self.rho_t = self.rho_scheduler.step()
        else:
            self.rho_t = 0.0
        return self.rho_t


    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
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
        
        # gradients without graph
        for p in params:
            p.grad = p.grad.detach()
                
    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        self.update_rho_t()
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = self.rho_t / (grad_norm + group["perturb_eps"])

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
    def _sync_gradient(self):
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
    def _sync_hessian(self):
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
    def step(self, iter_num, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        self.update_hessian_power()

        loss = None
        if closure is not None:
            loss = closure()

        self._sync_hessian()

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
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                                           
                exp_avg, exp_hessian_diag = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                if iter_num % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                     
                step_size = group['lr']
                step_size_neg = -step_size
                denom = (exp_hessian_diag**self.hessian_power_t).add_(group['eps'])

                # make update
                p.addcdiv_(exp_avg, denom, value=step_size_neg)
        
        return loss
