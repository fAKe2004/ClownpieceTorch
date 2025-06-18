from .clownpiece import *
from .utils import wrap_tuple
from .tensor import Tensor
# from .autograd import backward


__all__ = ['TensorBase', 'Tensor']

def ones(*args, **kwargs):
    return Tensor.ones(*args, **kwargs)

def ones_like(*args, **kwargs):
    return Tensor.ones_like(*args, **kwargs)

def zeros(*args, **kwargs):
    return Tensor.zeros(*args, **kwargs)

def zeros_like(*args, **kwargs):
    return Tensor.zeros_like(*args, **kwargs)