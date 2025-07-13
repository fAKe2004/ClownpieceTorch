from clownpiece.autograd import no_grad
from clownpiece.nn import Parameter
from clownpiece.nn.init import zeros_
from clownpiece.tensor import Tensor

from typing import List, Iterable, Dict, Any, Union

class Optimizer():
  
  param_groups: List[Dict[str, Any]]
  state: Dict[Parameter, Dict[str, Any]]
  defaults: Dict[str, Any]
  
  def __init__(self, parameters: Union[Iterable[Parameter], Iterable[Dict[str, Any]]], defaults: Dict[str, Any]):
    self.defaults = defaults
    self.param_groups = []
    self.state = {}
    
    param_groups = list(parameters)
    if len(param_groups) == 0:
        raise ValueError("optimizer got an empty parameter list")
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    for param_group in param_groups:
        self.add_param_group(param_group)
        
    self._assign_param_id()

  def add_param_group(self, param_group: Dict[str, Any]):
    """
    Adds a parameter group to the optimizer's param_groups.
    """
    # Add default values for any missing optimizer-specific options
    # In this base class, there are no defaults to add, but subclasses will use this.
    for k, v in self.defaults.items():
        param_group.setdefault(k, v)
        
    self.param_groups.append(param_group)
    
  def step(self):
    """
    Update the parameters based on the gradients.
    This method should be implemented by subclasses.
    """
    raise NotImplementedError("Optimizer step method not implemented")
  
  def zero_grad(self, set_to_None:bool=True):
    for group in self.param_groups:
      for param in group["params"]:
        if param.grad is None:
          continue
        
        if set_to_None:
          param.grad = None
        else:
          zeros_(param.grad)
      
  
  def _assign_param_id(self):
    idx = 0
    for param_group in self.param_groups:
      params = param_group.get('params', [])
      for param in params:
        if not isinstance(param, Parameter) or hasattr(param, 'param_id'):
          continue
        param.param_id = idx
        idx += 1
  
class SGD(Optimizer):
  """
  Stochastic Gradient Descent optimizer with momentum and weight decay.
  """
  def __init__(self, params, lr: float, momentum: float = 0.0, damping: float = 0.0, weight_decay: float = 0.0):
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if momentum < 0.0:
        raise ValueError(f"Invalid momentum value: {momentum}")
    if damping < 0.0:
        raise ValueError(f"Invalid damping value: {damping}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    # Store hyperparameters
    defaults = dict(lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay)
    super().__init__(params, defaults)

  def step(self):
    """
    Performs a single optimization step:
      1. Applies L2 weight decay: grad = grad + weight_decay * p
      2. Updates momentum buffer: buf = momentum * buf + grad
      3. Updates parameter: p = p - lr * buf
    """
    with no_grad():
      for group in self.param_groups:
        lr = group['lr']
        momentum = group['momentum']
        damping = group['damping']
        weight_decay = group['weight_decay']
        for p in group['params']:
          if p.grad is None:
            continue
          grad = p.grad
          # L2 regularization
          if weight_decay != 0:
            grad = grad + weight_decay * p
          # retrieve or init momentum buffer
          state = self.state.get(p.param_id)
          if state is None:
            buf = Tensor.zeros_like(p)
            state = {'momentum_buffer': buf}
            self.state[p.param_id] = state
          else:
            buf = state['momentum_buffer']
          # update buffer and parameters
          buf.copy_(momentum * buf + (1 - damping) * grad)
          p.copy_(p - lr * buf)

class Adam(Optimizer):
  """
  Implements the Adam algorithm.
  """
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if not 0.0 <= lr:
      raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    if not 0.0 <= weight_decay:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
      
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    super().__init__(params, defaults)

  def step(self):
    for group in self.param_groups:
      lr = group['lr']
      beta1, beta2 = group['betas']
      eps = group['eps']
      weight_decay = group['weight_decay']
      
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        
        if weight_decay != 0:
          grad = grad + weight_decay * p
        
        state = self.state.get(p.param_id)
        
        # State initialization
        if state is None:
          state = {}
          # Exponential moving average of gradient values
          state['exp_avg'] = Tensor.zeros_like(p)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = Tensor.zeros_like(p)
          # Step
          state['step'] = 0
          self.state[p.param_id] = state
        
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        state['step'] += 1
        
        # Decay the first and second moment running average coefficient
        exp_avg.copy_(beta1 * exp_avg + (1 - beta1) * grad)
        exp_avg_sq.copy_(beta2 * exp_avg_sq + (1 - beta2) * (grad * grad))
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        step_size = lr / bias_correction1
        
        # Update parameters
        denom = (exp_avg_sq / bias_correction2).sqrt() + eps
        p.copy_(p - step_size * (exp_avg / denom))
