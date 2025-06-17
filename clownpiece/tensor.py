from .clownpiece import TensorBase
import clownpiece as cp
from typing import TYPE_CHECKING, List, Optional, Union
import copy
import importlib

if TYPE_CHECKING:
  from clownpiece.autograd.autograd import Function

"""
  Autograd Utils (avoid circular import only)
"""
def is_grad_enabled():
  from clownpiece.autograd.no_grad import is_grad_enabled
  return is_grad_enabled()

def is_grad_enabled_with_params(*args):
  flatten_args = []
  for arg in args:
    if isinstance(arg, (list, tuple)):
      flatten_args.extend(arg)
    else:
      flatten_args.append(arg)
  
  return is_grad_enabled() and any(tensor.requires_grad for tensor in flatten_args if isinstance(tensor, Tensor))


"""
    Utils for Binding
"""

"""
    Wrap scalar args to singleton Tensor (requires_grad = False)
"""
def scalar_to_tensor(function):
    def wrapped_function(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, (int, float)):
                new_args.append(Tensor(arg, requires_grad=False))
            else:
                new_args.append(arg)
        return function(*new_args, **kwargs)
    return wrapped_function

"""
    Wrap around a Tensor operator that traces gradient.
    @arg: op_name: the name of the TensorBase method to call.
    @arg: Function_name: the name of the Function class to use for autograd.
"""
def tensor_op(op_name, Function_name):
    def decorator(function):
        def wrapped_function(self, *args, **kwargs):
            if not is_grad_enabled_with_params(*args):
                op = getattr(TensorBase, op_name)
                raw_results = op(self._cdata, *args, **kwargs)
                def TensorBase2Tensor(x):
                    return Tensor(x, requires_grad=False) if isinstance(x, TensorBase) else x
                if isinstance(raw_results, (list, tuple)):
                    return tuple(TensorBase2Tensor(x) for x in raw_results)
                else:
                    return TensorBase2Tensor(raw_results)
            module = importlib.import_module("clownpiece.autograd.function")
            FunctionClass = getattr(module, Function_name)
            return function(*args, **kwargs, FunctionClass=FunctionClass)
        return wrapped_function
    return decorator
  
"""
  Tensor Class 
"""


class Tensor():
  requires_grad: bool
  grad_fn: Optional["Function"]
  grad: Optional["Tensor"]
  requires_grad: bool
  output_nr: int
  
  """
    Initialization
  """
  def __init__(self, input_array, requires_grad: bool = None):
    print("Tensor.__new__ input type:", type(input_array))
    if isinstance(input_array, cp.TensorBase):
      self._cdata = input_array
    else:
      self._cdata = cp.TensorBase(input_array)
      
    self.requires_grad_(requires_grad)
    self.grad_fn = None
    self.grad = None
    self.output_nr = 0

  def __array_finalize__(self, obj):
    # if obj is None:
    #   return
    self.requires_grad = getattr(obj, 'requires_grad', False)
    self.grad_fn = getattr(obj, 'grad_fn', None)
    self.grad = getattr(obj, 'grad', None)
    self.output_nr = getattr(obj, 'output_nr', 0)

    # Ensure attributes are initialized if not present on obj
    if not hasattr(self, 'grad_fn'):
        self.grad_fn = None
    if not hasattr(self, 'grad'):
        self.grad = None
    if not hasattr(self, 'output_nr'):
        self.output_nr = 0
        
  """
    Other
  """
  def backward(self, grad: Optional["Tensor"]=None):
    from clownpiece.autograd.autograd import backward
    backward(self, grad)
      
  def requires_grad_(self, requires_grad:bool=None):
    if requires_grad is None:
      requires_grad = is_grad_enabled()
    self.requires_grad = requires_grad 
  
  @property
  def shape(self):
    # print("self._cdata.shape=", self._cdata.shape)
    return self._cdata.shape
  
  def __len__(self)->int:
    return cp.numel(self._cdata)
  

  def __getitem__(self, index):
    # 返回新的Python Tensor实例，包含底层切片TensorBase
    return Tensor(self[index], requires_grad=self.requires_grad)
      
  """
    Operator Binding for Autograd
  """
  
  """
    Part 1
  """
  @tensor_op('clone', 'Clone')
  def clone(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('contiguous', 'Contiguous')
  def contiguous(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('__getitem__', 'Subscriptor')
  def __getitem__(self, index, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, index)
  
  @tensor_op('item', 'Item')
  def item(self)->float:
    return self._cdata.item()
    # return FunctionClass().apply(self, index)
    
  """
    Part 2
  """
  @tensor_op('__neg__', 'Neg')
  def __neg__(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('sign', 'Sign')
  def sign(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('abs', 'Abs')
  def abs(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self) 
  
  @tensor_op('sin', 'Sin')
  def sin(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('cos', 'Cos')
  def cos(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('tanh', 'Tanh')
  def tanh(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('clamp', 'Clamp')
  def clamp(self, min_val, max_val, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, min_val, max_val)


  @tensor_op('log', 'Log')
  def log(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('exp', 'Exp')
  def exp(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('pow', 'Pow')
  def pow(self, exponent, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, exponent)

  @tensor_op('sqrt', 'Sqrt')
  def sqrt(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  """
    Part 3
  """
  
  @tensor_op('__add__', 'Add')
  @scalar_to_tensor
  def __add__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
  @tensor_op('__radd__', 'Add')
  @scalar_to_tensor
  def __radd__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)

  @tensor_op('__sub__', 'Sub')
  @scalar_to_tensor
  def __sub__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
  @tensor_op('__rsub__', 'Sub')
  @scalar_to_tensor
  def __rsub__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)
  
  @tensor_op('__mul__', 'Mul')
  @scalar_to_tensor
  def __mul__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
    
  @tensor_op('__rmul__', 'Mul')
  @scalar_to_tensor
  def __rmul__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)

  @tensor_op('__truediv__', 'Div')
  @scalar_to_tensor
  def __truediv__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
  @tensor_op('__rtruediv__', 'Div')
  @scalar_to_tensor
  def __rtruediv__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)
  
  """
  Part 4
  """
  @tensor_op('matmul', 'MatMul')
  def matmul(self, other, FunctionClass=None)->"Tensor":
    if not isinstance(other, Tensor):
      raise TypeError(f"Expected Tensor, got {type(other).__name__}")
    
    return FunctionClass().apply(self, other)
  
  def __matmul__(self, other)->"Tensor":    
    return self.matmul(other)
  
  def __rmatmul__(self, other)->"Tensor":
    if not isinstance(other, Tensor):
      raise TypeError(f"Expected Tensor, got {type(other).__name__}")
    return other.matmul(self)
  
  """
  Part 5
  """
  @tensor_op('sum', 'Sum')
  def sum(self, dim=None, keepdims=False, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, dim, keepdims)
  
  @tensor_op('max', 'Max')
  def max(self, dim=None, keepdims=False, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, dim, keepdims)
  
  @tensor_op('softmax', 'Softmax')
  def softmax(self, dim=None, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, dim)
  
  """
  Part 6
  """
  
  @tensor_op('permute', 'Permute')
  def permute(self, perm: List[int], FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, perm)

  @tensor_op('transpose', 'Transpose')
  def transpose(self, dim0: int, dim1: int, FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, dim0, dim1)

  @tensor_op('reshape', 'Reshape')
  def reshape(self, shape: List[int], FunctionClass=None) -> "Tensor":
      tb = self._cdata.reshape(shape)
      return Tensor(tb, requires_grad=self.requires_grad)

  @tensor_op('view', 'View')
  def view(self, shape: List[int], FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, shape)

  @tensor_op('narrow', 'Narrow')
  def narrow(self, dim: int, start: int, length: int, FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, dim, start, length)

  @tensor_op('chunk', 'Chunk')
  def chunk(self, chunks: int, dim: int = 0, FunctionClass=None) -> List["Tensor"]:
      return FunctionClass().apply(self, chunks, dim)

  @tensor_op('split', 'Split')
  def split(self, split: Union[int, List[int]], dim: int = 0, FunctionClass=None) -> List["Tensor"]:
      return FunctionClass().apply(self, split, dim)

  @tensor_op('stack', 'Stack')
  @staticmethod
  def stack(inputs: List["Tensor"], dim: int = 0, FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(*inputs, dim=dim)

  @tensor_op('cat', 'Cat')
  @staticmethod
  def cat(inputs: List["Tensor"], dim: int = 0, FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(*inputs, dim=dim)

  @tensor_op('squeeze', 'Squeeze')
  def squeeze(self, dim: int = 0, FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, dim)

  @tensor_op('unsqueeze', 'Unsqueeze')
  def unsqueeze(self, dim: int = 0, FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, dim)

  @tensor_op('broadcast_to', 'BroadcastTo')
  def broadcast_to(self, shape: List[int], FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(self, shape)

  @tensor_op('broadcast', 'Broadcast')
  @staticmethod
  def broadcast(inputs: List["Tensor"], FunctionClass=None) -> "Tensor":
      return FunctionClass().apply(*inputs)
  
  """
  STR
  """
  
  def __repr__(self):    
    grad_fn_name = self.grad_fn.__class__.__name__ if self.grad_fn else None
    return f"Tensor({self._cdata.__repr__()}, requires_grad={self.requires_grad}, grad_fn={grad_fn_name})"
  
def stack(inputs: List[Tensor], dim: int = 0) -> Tensor:
  return Tensor.stack(inputs, dim)

def cat(inputs: List[Tensor], dim: int = 0) -> Tensor:
  return Tensor.cat(inputs, dim)

def broadcast(inputs: List[Tensor]) -> Tensor:
  return Tensor.broadcast(inputs)

"""
  Constructors
"""
  
def empty(shape, requires_grad: bool = False) -> Tensor:
  return Tensor(cp.empty(shape), requires_grad=requires_grad)

def empty_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
  return Tensor(cp.empty_like(tensor), requires_grad=requires_grad)

def zeros(shape, requires_grad: bool = False) -> Tensor:
  return Tensor(cp.zeros(shape), requires_grad=requires_grad)

def zeros_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
  return Tensor(cp.zeros_like(tensor), requires_grad=requires_grad)

def ones(shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)
    return Tensor(cp.ones(shape), requires_grad=requires_grad)

def ones_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
  return Tensor(cp.ones_like(tensor), requires_grad=requires_grad)
