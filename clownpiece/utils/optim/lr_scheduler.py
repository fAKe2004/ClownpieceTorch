from .optimizer import Optimizer
from typing import List

class LRScheduler:
    """
    Base class for learning rate schedulers.
    """
    
    def __init__(self, optimizer: Optimizer, last_epoch=-1):
        self.optimizer = optimizer

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self):
        """
        Compute learning rate. This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self, epoch=None):
        """
        Update the learning rate based on the current epoch or iteration.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """
        Get the last computed learning rates for each parameter group.
        """
        return [group['lr'] for group in self.optimizer.param_groups]
      
class LambdaLR(LRScheduler):
    """
    Lambda learning rate scheduler.
    Applies a user-defined function to the learning rate.
    """
    
    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.lr_lambda = lr_lambda

    def get_lr(self):
        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
    
class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.
    Multiplies the learning rate by a factor every epoch.
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float = 0.1, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        steps = self.last_epoch
        return [base_lr * (self.gamma ** steps) for base_lr in self.base_lrs]
        
class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    Decreases the learning rate by a factor every `step_size` epochs.
    """
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        steps = self.last_epoch // self.step_size
        return [base_lr * (self.gamma ** steps) for base_lr in self.base_lrs]
    
