# Acknowledgement: This code is based on https://github.com/MarlonBecker/MSAM/blob/main/optimizer/adamW_msam.py

from typing import List
from torch import Tensor
import torch
import math

# cf. https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
class MSASSHA(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 0.15,
            betas: float = (0.9, 0.999),
            weight_decay: float = 1e-2,
            lazy_hessian: int = 10,
            rho: float = 0.3,
            n_samples: int = 1,
            eps: float = 1e-4,
            hessian_power: int = 1,
            seed: int = 0,
            maximize: bool = False
            ):
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            rho=rho,
            eps=eps,
            maximize=maximize
        )

        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.seed = seed
        self.hessian_power = hessian_power

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(self.seed)

        super(MSASSHA, self).__init__(params, defaults)

        # init momentum buffer to zeros
        # needed to make implementation of first ascent step cleaner (before SGD.step() was ever called)

        for p in self.get_params():
            p.hess = 0.0
            if self.track_hessian:
                p.real_hess = 0.0

            state = self.state[p]
            state["hessian_step"] = 0
            
        for group in self.param_groups:
            group["norm_factor"] = [0,]

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)
    

    def zero_hessian(self):
        """
        Zeros out the accumulated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian_step"] % self.lazy_hessian == 0:
                p.hess.zero_()


    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian_step"] % self.lazy_hessian == 0:  # compute a new Hessian per `lazy_hessian` step
                params.append(p)
            self.state[p]["hessian_step"] += 1

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
    def step(self, closure=None):
        """ Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) > 1:
            raise RuntimeError("only one parameter group supported atm for SAMANDA_MSAM")

        group = self.param_groups[0]
        params_with_grad = []
        grads = []
        hesses = []
        grad_momentums = []
        hess_momentums = []
        state_steps = []
    
        beta1, beta2 = group['betas']

        self.zero_hessian()
        self.set_hessian()
     
        for p in group['params']:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError('Msassha does not support sparse gradients')
            
            grads.append(p.grad)
            hesses.append(p.hess)

            state = self.state[p]
            # State initialization
            if len(state) == 1:
                state['step'] = 0
                state['grad_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['hess_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            grad_momentums.append(state['grad_momentum'])
            hess_momentums.append(state['hess_momentum'])

            # update the steps for each param group update
            state['step'] += 1
            # record the step after step update
            state_steps.append(state['step'])

        samanda_msam(params_with_grad,
                     grads,
                     hesses,
                     grad_momentums,
                     hess_momentums,
                     state_steps,
                     beta1=beta1,
                     beta2=beta2,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     lazy_hessian=self.lazy_hessian,
                     rho=group['rho'],
                     norm_factor=group['norm_factor'],
                     eps=group['eps']
                     )
        
        return loss


    @torch.no_grad()
    def move_up_to_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "grad_momentum" in self.state[p]:
                    p.sub_(self.state[p]["grad_momentum"], alpha=group["norm_factor"][0])


    @torch.no_grad()
    def move_back_from_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "grad_momentum" in self.state[p]:
                    p.add_(self.state[p]["grad_momentum"], alpha=group["norm_factor"][0])

bias_correction2 = 0
def samanda_msam(params: List[Tensor],
          grads: List[Tensor],
          hesses: List[Tensor],
          grad_momentums: List[Tensor],
          hess_momentums: List[Tensor],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          lazy_hessian: int,
          rho:float,
          norm_factor: list,
          eps: float
          ):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    
    for i, param in enumerate(params):
        grad = grads[i]
        hess = hesses[i]
        grad_momentum = grad_momentums[i]
        hess_momentum = hess_momentums[i]
        step = state_steps[i]

        # remove last perturbation (descent) w_t <- \tilde{w_t} + rho*m_t/||m_t|| 
        param.add_(grad_momentum, alpha=norm_factor[0])

        # weight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        grad_momentum.mul_(beta1).add_(grad, alpha=1-beta1)
        bias_correction1 = 1 - beta1 ** step

        if (step-1) % lazy_hessian == 0:
            hess_momentum.mul_(beta2).add_(hess.abs(), alpha=1-beta2)
            global bias_correction2
            bias_correction2 = 1 - beta2 ** step
        
        denom = ((hess_momentum ** self.hessian_power) / (bias_correction2 ** self.hessian_power)).add_(eps)

        step_size = lr / bias_correction1

        # make update
        param.addcdiv_(grad_momentum, denom, value=-step_size)

    #calculate ascent step norm
    ascent_norm = torch.norm(
                torch.stack([
                    grad_momentum.norm(p=2)
                    for grad_momentum in grad_momentums
                ]),
                p=2
        )
    norm_factor[0] = 1/(ascent_norm+1e-12) * rho
    
    # perturb for next iteration (ascent)
    for i, param in enumerate(params):
        param.sub_(grad_momentums[i], alpha = norm_factor[0])

