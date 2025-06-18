import clownpiece as cp
from clownpiece import TensorBaseImpl

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
  Tensor Base Class
"""

class TensorBase:
  def __init__(self, array=None, requires_grad=None):
    if isinstance(array, TensorBase):
      self._impl = array._impl
    elif isinstance(array, cp.TensorBaseImpl):
      self._impl = array
    elif array is not None:
      self._impl = cp.TensorBaseImpl(array)
    else:
      self._impl = cp.TensorBaseImpl()
    # 需要时可以保存 requires_grad，但通常子类才用

  @classmethod
  def ones(cls, shape, **kwargs):
    # 只把 shape/dtype 传给 C++，其余参数传到 Python 层
    print("Base Creating ones with shape:", shape)
    impl = cp.TensorBaseImpl.ones(shape)
    return cls(impl, **kwargs)
  
  @classmethod
  def ones_like(cls, tensor, **kwargs):
    # 只把 shape/dtype 传给 C++，其余参数传到 Python 层
    print("Base Creating ones_like with tensor:", tensor)
    impl = cp.TensorBaseImpl.ones_like(tensor._impl)
    return cls(impl, **kwargs)
  
  @classmethod
  def zeros(cls, shape, **kwargs):
    # 只把 shape/dtype 传给 C++，其余参数传到 Python 层
    print("Base Creating zeros with shape:", shape)
    impl = cp.TensorBaseImpl.zeros(shape)
    return cls(impl, **kwargs)
  
  @classmethod
  def zeros_like(cls, tensor, **kwargs):
    # 只把 shape/dtype 传给 C++，其余参数传到 Python 层
    print("Base Creating zeros_like with tensor:", tensor)
    impl = cp.TensorBaseImpl.zeros_like(tensor._impl)
    return cls(impl, **kwargs)
  

  @property
  def shape(self):
    return self._impl.shape
      
  def reshape(self, new_shape):
    # 假设 C++ 有 reshape()
    reshaped_impl = self._impl.reshape(new_shape)
    return self.__class__(reshaped_impl)

  def __getitem__(self, idx):
    # 假设 C++ 的 TensorBaseImpl 支持 __getitem__
    item = self._impl[idx]
    # 若返回子 Tensor，也要包装
    if isinstance(item, cp.TensorBaseImpl):
      return self.__class__(item)
    return item

  def __setitem__(self, idx, value):
    if isinstance(value, TensorBase):
      self._impl[idx] = value._impl
    else:
      self._impl[idx] = value
  
  def item(self):
    return self._impl.item()

  def __len__(self):
      # 通常返回第一个维度的长度，等价于 shape[0]
      return cp.numel(self._impl)
  def clone(self):
    # _impl 是 C++ 对象，应该有clone方法（或copy）
    return type(self)(self._impl.clone())
  
  def __repr__(self):
    # return reqiures_grad = getattr(self, 'requires_grad', None)
    return f"({self._impl}, requires_grad={self.requires_grad})"
#   def contiguous(self):
#     return TensorBase(np.ascontiguousarray(self))
#   def transpose(self, dim0=-1, dim1=-2):
#     return TensorBase(super().swapaxes(dim0, dim1))
#   def copy_(self, other):
#     self[:] = other
#   def scatter_(self, dim, index, src):
#     data = self.copy()
#     shape_dim = data.shape[dim]
#     data = data.swapaxes(-1, dim)
#     data = data.reshape((-1, shape_dim))
#     index = index.reshape(-1)
#     src = src.reshape(-1)
#     for i in range(len(index)):
#       data[i][index[i]] = src[i]
#     data = data.swapaxes(-1, dim)
#     data = data.reshape(self.shape)
#     self.copy_(data)
    
#   def unsqueeze(self, dim=0):
#     return TensorBase(np.expand_dims(self, axis=dim))
  
#   def __neg__(self): return TensorBase(super().__neg__())
#   def sign(self): return TensorBase(np.sign(self))
#   def abs(self): return TensorBase(np.abs(self))
#   def sin(self): return TensorBase(np.sin(self))
#   def cos(self): return TensorBase(np.cos(self))
#   def tanh(self): return TensorBase(np.tanh(self))
#   def clamp(self, min_val, max_val): return TensorBase(np.clip(self, min_val, max_val))
#   def log(self): return TensorBase(np.log(self))
#   def exp(self): return TensorBase(np.exp(self))
#   def pow(self, exponent): return TensorBase(np.power(self, exponent))
#   def sqrt(self): return TensorBase(np.sqrt(self))
  def __add__(self, other):
    if isinstance(other, TensorBase):
      return TensorBase(self._impl + other._impl)
    else:
      return TensorBase(self._impl + other)

#   def matmul(self, other): return TensorBase(np.matmul(self, other))
  
#   def dim(self): return len(self.shape)
  
#   def sum(self, dim=-1, keepdims=False):
#     return TensorBase(super(TensorBase, self).sum(axis=dim, keepdims=keepdims))
  
#   def max(self, dim=-1, keepdims=False):
#     return TensorBase(super(TensorBase, self).max(axis=dim, keepdims=keepdims)), TensorBase(np.argmax(self, axis=dim, keepdims=keepdims))
#   def softmax(self, dim=-1):
#     exp_tensor = np.exp(self)
#     sum_tensor = exp_tensor.sum(dim=dim, keepdims=True)
#     result = TensorBase(exp_tensor / sum_tensor)
#     return result
  
#   def permute(self, perm: List[int]):
#     return TensorBase(np.transpose(self, axes=perm))
#   def reshape(self, shape: List[int]):
#     return TensorBase(np.reshape(self, shape))
#   def view(self, shape: List[int]):
#     return TensorBase(np.reshape(self, shape))
#   def narrow(self, dim: int, start: int, length: int):
#     slices = [slice(None)] * self.dim()
#     slices[dim] = slice(start, start + length)
#     return TensorBase(self[tuple(slices)])
#   def chunk(self, chunks: int, dim: int = 0):
#     split_sections = [(self.shape[dim] + chunks - 1) // chunks] * chunks
#     split_indices = [sum(split_sections[:i]) for i in range(1, len(split_sections))] # omit last one
#     splitted = np.array_split(self, split_indices, axis=dim)
#     return [TensorBase(t) for t in splitted]
  
#   def split(self, split: Union[int, List[int]], dim: int = 0):
#     if isinstance(split, list):
#       split = [sum(split[:i]) for i in range(1, len(split))] # omit last one
#     return [TensorBase(t) for t in np.array_split(self, split, axis=dim)]
  
#   @staticmethod
#   def cat(inputs: List["TensorBase"], dim: int = 0):
#     return TensorBase(np.concatenate([np.asarray(t) for t in inputs], axis=dim))
#   @staticmethod
#   def stack(inputs: List["TensorBase"], dim: int = 0):
#     return TensorBase(np.stack([np.asarray(t) for t in inputs], axis=dim))
#   def broadcast_to(self, shape):
#     return TensorBase(np.broadcast_to(self, shape))
#   @staticmethod
#   def broadcast(inputs: List["TensorBase"]):
#     outputs = np.broadcast_arrays(*inputs)
#     outputs = [TensorBase(o) for o in outputs]
#     return outputs
  

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
    def wrapped_function(*args, **kwargs):
    
      if not is_grad_enabled_with_params(*args):
        op = getattr(TensorBase, op_name)
        raw_results = op(*args, **kwargs)
        
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

class Tensor(TensorBase):
  requires_grad: bool
  grad_fn: Optional["Function"]
  grad: Optional["Tensor"]
  requires_grad: bool
  output_nr: int
  
  def __init__(self, array=None, requires_grad=None):
    super().__init__(array)
    self.requires_grad = requires_grad
    self.grad = None
    self.grad_fn = None
    self.output_nr = 0

  # @classmethod
  # def ones(cls, shape, requires_grad=None):
  #   print("got shape=", shape)
  #   tensor = super().ones(shape)
  #   print("got tensor type=", type(tensor))
  #   tensor.requires_grad = requires_grad
  #   return tensor

#   def __array_finalize__(self, obj):
#     # if obj is None:
#     #   return
#     self.requires_grad = getattr(obj, 'requires_grad', False)
#     self.grad_fn = getattr(obj, 'grad_fn', None)
#     self.grad = getattr(obj, 'grad', None)
#     self.output_nr = getattr(obj, 'output_nr', 0)

#     # Ensure attributes are initialized if not present on obj
#     if not hasattr(self, 'grad_fn'):
#         self.grad_fn = None
#     if not hasattr(self, 'grad'):
#         self.grad = None
#     if not hasattr(self, 'output_nr'):
#         self.output_nr = 0
        
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
  

      
  """
    Operator Binding for Autograd
  """
  
  """
    Part 1
  """
  @tensor_op('clone', 'Clone')
  def clone(self, FunctionClass=None)->"Tensor":
    t = FunctionClass().apply(self)
    t.requires_grad = self.requires_grad
    t.grad_fn = self.grad_fn
    t.grad = self.grad
    t.output_nr = self.output_nr
    return t

  
  # @tensor_op('contiguous', 'Contiguous')
  # def contiguous(self, FunctionClass=None)->"Tensor":
  #   return FunctionClass().apply(self)
  
  # @tensor_op('__getitem__', 'Subscriptor')
  # def __getitem__(self, index, FunctionClass=None)->"Tensor":
  #   return FunctionClass().apply(self, index)
    
#   """
#     Part 2
#   """
#   @tensor_op('__neg__', 'Neg')
#   def __neg__(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('sign', 'Sign')
#   def sign(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('abs', 'Abs')
#   def abs(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self) 
  
#   @tensor_op('sin', 'Sin')
#   def sin(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('cos', 'Cos')
#   def cos(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('tanh', 'Tanh')
#   def tanh(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('clamp', 'Clamp')
#   def clamp(self, min_val, max_val, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, min_val, max_val)


#   @tensor_op('log', 'Log')
#   def log(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('exp', 'Exp')
#   def exp(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
#   @tensor_op('pow', 'Pow')
#   def pow(self, exponent, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, exponent)

#   @tensor_op('sqrt', 'Sqrt')
#   def sqrt(self, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self)
  
  """
    Part 3
  """
  
  @tensor_op('__add__', 'Add')
  @scalar_to_tensor
  def __add__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
#   @tensor_op('__radd__', 'Add')
#   @scalar_to_tensor
#   def __radd__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(other, self)

#   @tensor_op('__sub__', 'Sub')
#   @scalar_to_tensor
#   def __sub__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, other)
  
#   @tensor_op('__rsub__', 'Sub')
#   @scalar_to_tensor
#   def __rsub__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(other, self)
  
#   @tensor_op('__mul__', 'Mul')
#   @scalar_to_tensor
#   def __mul__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, other)
    
#   @tensor_op('__rmul__', 'Mul')
#   @scalar_to_tensor
#   def __rmul__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(other, self)

#   @tensor_op('__truediv__', 'Div')
#   @scalar_to_tensor
#   def __truediv__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, other)
  
#   @tensor_op('__rtruediv__', 'Div')
#   @scalar_to_tensor
#   def __rtruediv__(self, other, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(other, self)
  
#   """
#   Part 4
#   """
#   @tensor_op('matmul', 'MatMul')
#   def matmul(self, other, FunctionClass=None)->"Tensor":
#     if not isinstance(other, Tensor):
#       raise TypeError(f"Expected Tensor, got {type(other).__name__}")
    
#     return FunctionClass().apply(self, other)
  
#   def __matmul__(self, other)->"Tensor":    
#     return self.matmul(other)
  
#   def __rmatmul__(self, other)->"Tensor":
#     if not isinstance(other, Tensor):
#       raise TypeError(f"Expected Tensor, got {type(other).__name__}")
#     return other.matmul(self)
  
#   """
#   Part 5
#   """
#   @tensor_op('sum', 'Sum')
#   def sum(self, dim=None, keepdims=False, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, dim, keepdims)
  
#   @tensor_op('max', 'Max')
#   def max(self, dim=None, keepdims=False, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, dim, keepdims)
  
#   @tensor_op('softmax', 'Softmax')
#   def softmax(self, dim=None, FunctionClass=None)->"Tensor":
#     return FunctionClass().apply(self, dim)
  
#   """
#   Part 6
#   """
  
#   @tensor_op('permute', 'Permute')
#   def permute(self, perm: List[int], FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, perm)

#   @tensor_op('transpose', 'Transpose')
#   def transpose(self, dim0: int, dim1: int, FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, dim0, dim1)

#   @tensor_op('reshape', 'Reshape')
#   def reshape(self, shape: List[int], FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, shape)

#   @tensor_op('view', 'View')
#   def view(self, shape: List[int], FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, shape)

#   @tensor_op('narrow', 'Narrow')
#   def narrow(self, dim: int, start: int, length: int, FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, dim, start, length)

#   @tensor_op('chunk', 'Chunk')
#   def chunk(self, chunks: int, dim: int = 0, FunctionClass=None) -> List["Tensor"]:
#       return FunctionClass().apply(self, chunks, dim)

#   @tensor_op('split', 'Split')
#   def split(self, split: Union[int, List[int]], dim: int = 0, FunctionClass=None) -> List["Tensor"]:
#       return FunctionClass().apply(self, split, dim)

#   @tensor_op('stack', 'Stack')
#   @staticmethod
#   def stack(inputs: List["Tensor"], dim: int = 0, FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(*inputs, dim=dim)

#   @tensor_op('cat', 'Cat')
#   @staticmethod
#   def cat(inputs: List["Tensor"], dim: int = 0, FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(*inputs, dim=dim)

#   @tensor_op('squeeze', 'Squeeze')
#   def squeeze(self, dim: int = 0, FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, dim)

#   @tensor_op('unsqueeze', 'Unsqueeze')
#   def unsqueeze(self, dim: int = 0, FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, dim)

#   @tensor_op('broadcast_to', 'BroadcastTo')
#   def broadcast_to(self, shape: List[int], FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(self, shape)

#   @tensor_op('broadcast', 'Broadcast')
#   @staticmethod
#   def broadcast(inputs: List["Tensor"], FunctionClass=None) -> "Tensor":
#       return FunctionClass().apply(*inputs)
  
#   """
#   STR
#   """
  
#   def __repr__(self):    
#     grad_fn_name = self.grad_fn.__class__.__name__ if self.grad_fn else None
#     return f"Tensor({super().__repr__()}, requires_grad={self.requires_grad}, grad_fn={grad_fn_name})"
  
# def stack(inputs: List[Tensor], dim: int = 0) -> Tensor:
#   return Tensor.stack(inputs, dim)

# def cat(inputs: List[Tensor], dim: int = 0) -> Tensor:
#   return Tensor.cat(inputs, dim)

# def broadcast(inputs: List[Tensor]) -> Tensor:
#   return Tensor.broadcast(inputs)

# """
#   Constructors
# """
  
# def empty(shape, requires_grad: bool = False) -> Tensor:
#   return Tensor(np.empty(shape), requires_grad=requires_grad)

# def empty_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
#   return Tensor(np.empty_like(tensor), requires_grad=requires_grad)

# def zeros(shape, requires_grad: bool = False) -> Tensor:
#   return Tensor(np.zeros(shape), requires_grad=requires_grad)

# def zeros_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
#   return Tensor(np.zeros_like(tensor), requires_grad=requires_grad)

# def ones(shape, requires_grad: bool = False) -> Tensor:
#   return Tensor(np.ones(shape), requires_grad=requires_grad)

# def ones_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
#   return Tensor(np.ones_like(tensor), requires_grad=requires_grad)

def zeros(shape, dtype=None, requires_grad=None):
  return Tensor.zeros(shape, requires_grad=requires_grad)

def zeros_like(tensor, dtype=None, requires_grad=None):
  # 假设Tensor有shape属性
  return Tensor.zeros(tensor.shape, requires_grad=requires_grad)

def ones(shape, dtype=None, requires_grad=None):
  return Tensor.ones(shape, requires_grad=requires_grad)

def ones_like(tensor, dtype=None, requires_grad=None):
  # 假设Tensor有shape属性
  return Tensor.ones(tensor.shape, requires_grad=requires_grad)