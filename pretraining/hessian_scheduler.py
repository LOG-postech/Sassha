import math
import numpy as np

class ProportionScheduler:
    def __init__(self, get_lr, max_lr, min_lr, max_value, min_value, t=0):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = t    
        self.get_lr = get_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value
        
        assert (max_lr > min_lr) or ((max_lr==min_lr) and (max_value==min_value)), "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
        "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
        "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."
    
        assert max_value >= min_value
        
        #self.step() # take 1 step during initialization to get self._last_lr
    
    def lr(self):
        return self._last_lr[0]
                
    def step(self):
        lr = self.get_lr(self.t)
        self.t += 1
            
        if self.max_lr > self.min_lr:
            value = self.max_value - (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value
        
        self._last_lr = [value]
        return value    

class CustomScheduler:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, t=0, warmup_steps=0, optimizer=None):
        super(CustomScheduler, self).__init__()
        self.t = t
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max
        
        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]
                
        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.max_value
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value
                
        self._last_lr = [value]
        return value
    
    def step_func(self):
        pass
    
    def lr(self):
        return self._last_lr[0]


class CustomConstant(CustomScheduler):
    def step_func(self):
        value = self.min_value
        return value
    
    
class CustomLinear(CustomScheduler):    
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (self.t - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        return value
    
class CustomCosine(CustomScheduler):
    def step_func(self):
        phase = (self.t-self.warmup_steps) / (self.total_steps-self.warmup_steps) * math.pi
        value = self.min_value + (self.max_value-self.min_value) * (np.cos(phase) + 1.) / 2.0
        return value
    