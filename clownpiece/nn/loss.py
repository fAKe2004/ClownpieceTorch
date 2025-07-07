# MSE, CrossEntropy

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

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
      if logits.shape[:-1] != target.shape:
          raise ValueError(f"logits batch size {logits.shape[:-1]} does not match target batch size {target.shape}")

      logits = logits.reshape((-1, logits.shape[-1]))
      target = target.reshape(-1)
      
      # probs = logits.softmax(dim=-1).clamp(self.eps, 1)
      # log_probs = probs.log()
      log_probs = logits - logits.exp().sum(dim=-1, keepdims=True).log()
      
      selected_list = []
      for i in range(logits.shape[0]):
          index = round(target[i].item())
          selected = log_probs[i, index]
          selected_list += [selected]
          
      return -sum(selected_list) / logits.shape[0]
  
