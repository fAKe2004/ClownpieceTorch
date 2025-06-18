from .clownpiece import Tensor, TensorBase

x = Tensor.ones((2, 3))
print("x type:", type(x))
print("x is Tensor?", isinstance(x, Tensor))
print("x is TensorBase?", isinstance(x, TensorBase))
print("x._impl type:", type(x._impl))