from clownpiece.tensor import Tensor
from clownpiece.autograd.no_grad import no_grad

from typing import Callable
import random
import math

_gain_lookup_table = {
  "linear": 1.0,
  "idenity": 1.0,
  "sigmoid": 1.0,
  "tanh": 5/3,
  "relu": math.sqrt(2),
  "leaky_relu": lambda a: math.sqrt(2 / (1 + a * a)),
  "selu": 3/4
}

def calcuate_gain(nonlinearity: str, a: float = 0) -> float:
  nonlinearity = nonlinearity.lower()

  if nonlinearity not in _gain_lookup_table:
    raise KeyError(f"Unkown nonlinearity: {nonlinearity}, choices are {list(_gain_lookup_table.keys())}")
  
  value = _gain_lookup_table[nonlinearity]
  if nonlinearity == "leaky_relu":
    return value(a)
  else:
    return value
  
def _no_grad_init(func):
    def wrapper(*args, **kwargs):
        with no_grad():
            return func(*args, **kwargs)
    return wrapper  
  
def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if tensor.dim() == 2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = len(tensor) // (num_output_fmaps * num_input_fmaps)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out

@_no_grad_init
def _init_with_generator(
    tensor: Tensor, 
    generator: Callable[[], float]
  ):
  
  for i in range(len(tensor)):
    tensor._impl.change_data_at(i, generator())
  
def constants_(tensor: Tensor, value: float):
  _init_with_generator(tensor, lambda: value)
    
def zeros_(tensor: Tensor):
  _init_with_generator(tensor, lambda: 0.0)
  
def ones_(tensor: Tensor):
  _init_with_generator(tensor, lambda: 1.0)
  
def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
  _init_with_generator(tensor, lambda: random.gauss(mean, std))

def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0):
  _init_with_generator(tensor, lambda: random.uniform(low, high))
  
def xavier_uniform_(tensor: Tensor, gain: float = 1.0):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  std = gain * math.sqrt(2.0 / (fan_in + fan_out))
  a = math.sqrt(3.0) * std
  uniform_(tensor, -a, a)

def xavier_normal_(tensor: Tensor, gain: float = 1.0):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  std = gain * math.sqrt(2.0 / (fan_in + fan_out))
  normal_(tensor, 0.0, std)

def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
  if mode not in ["fan_in", "fan_out"]:
      raise ValueError(f"Unsupported mode: {mode}")
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  fan = fan_in if mode == "fan_in" else fan_out
  gain = calcuate_gain(nonlinearity, a)
  std = gain / math.sqrt(fan)
  bound = math.sqrt(3.0) * std
  uniform_(tensor, -bound, bound)

def kaiming_normal_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
  if mode not in ["fan_in", "fan_out"]:
      raise ValueError(f"Unsupported mode: {mode}")
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  fan = fan_in if mode == "fan_in" else fan_out
  gain = calcuate_gain(nonlinearity, a)
  std = gain / math.sqrt(fan)
  normal_(tensor, 0.0, std)