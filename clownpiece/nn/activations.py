from ..tensor import Tensor
from .module import Module
from ..autograd.function import Sigmoid as SigmoidFunction
from ..autograd.function import ReLU as ReLUFunction
from ..autograd.function import Tanh as TanhFunction


class Sigmoid(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: Tensor) -> Tensor:
    return SigmoidFunction.apply(x)

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ReLUFunction.apply(x)

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return TanhFunction.apply(x)
      
class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return x * (x > 0) + self.negative_slope * x * (x <= 0)