import clownpiece
from functools import wraps


# wrap x into tuple if it's not already
def wrap_tuple(x):
  return (x,) if not isinstance(x, (list, tuple)) else tuple(x)

# Decorator to use C++ tensor implementation for a function
def use_cpp_tensor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_tensor = clownpiece.Tensor
        clownpiece.Tensor = clownpiece.TensorBase
        try:
            return func(*args, **kwargs)
        finally:
            clownpiece.Tensor = old_tensor
    return wrapper
