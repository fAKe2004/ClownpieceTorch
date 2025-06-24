# MSE, CrossEntropy, etc.

from clownpiece.nn.module import Module
from clownpiece import Tensor

class MSELoss(Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    if input.shape != target.shape:
      raise ValueError(f"Input shape {input.shape} does not match target shape {target.shape}")
    diff = input - target
    return (diff.pow(2)).reshape(-1).mean(-1)
  
class CrossEntropyLoss(Module):
    def __init__(self):
      super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
      if input.ndim != 2 or target.ndim != 1:
          raise ValueError("Input must be 2D and target must be 1D")
      if input.shape[0] != target.shape[0]:
          raise ValueError(f"Input batch size {input.shape[0]} does not match target batch size {target.shape[0]}")
      
      log_probs = input.log().softmax(dim=-1)
      selected = []
      for i in range(input.shape[0]):
          selected += [log_probs[i, target[i]]]
          
      return -sum(selected) / input.shape[0]
  
