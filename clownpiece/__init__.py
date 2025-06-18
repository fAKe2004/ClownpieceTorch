from .clownpiece import *
from .utils import wrap_tuple
from .tensor import Tensor
# from .autograd import backward


__all__ = ['TensorBase', 'Tensor']

def stack(*args, **kwargs):
    return Tensor.stack(*args, **kwargs)

def cat(*args, **kwargs):
    return Tensor.cat(*args, **kwargs)

def broadcast(*args, **kwargs):
    return Tensor.broadcast(*args, **kwargs)

def ones(*args, **kwargs):
    return Tensor.ones(*args, **kwargs)

def ones_like(*args, **kwargs):
    return Tensor.ones_like(*args, **kwargs)

def zeros(*args, **kwargs):
    return Tensor.zeros(*args, **kwargs)

def zeros_like(*args, **kwargs):
    return Tensor.zeros_like(*args, **kwargs)

def empty(*args, **kwargs):
    return Tensor.empty(*args, **kwargs)

def empty_like(*args, **kwargs):
    return Tensor.empty_like(*args, **kwargs)