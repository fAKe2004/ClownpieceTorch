"""
    Implement Various Functions
"""

from typing import List, Union
import copy

from clownpiece.tensor import Tensor, zeros, zeros_like
from clownpiece.autograd.autograd import Node, Edge
from clownpiece.autograd.no_grad import no_grad
from clownpiece.utils import wrap_tuple


class Context():
    def __init__(self):
        self.saved_tensors = []
        
    def save_for_backward(self, *args) -> None:
        self.saved_tensors.extend(
            [self.repack_tensor(tensor) for tensor in args if isinstance(tensor, Tensor)]
        )
        
    def get_saved_tensors(self) -> List[Tensor]:
        return self.saved_tensors
    
    @staticmethod
    def repack_tensor(tensor: Tensor):
        # avoid cyclic reference
        if isinstance(tensor, Tensor):
            return copy.copy(tensor) # shallow copy
        else:
            return tensor
    
class Function(Node):
    """
    Base class for all functions.
    """
    ctx: Context
    
    def __init__(self):
        super().__init__()
        self.ctx = None
        
    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented")

    @staticmethod
    def backward(ctx: Context, *args):
        raise NotImplementedError("Backward method not implemented")    
    
    def apply(self, *args, **kwargs):
        self.next_edges = [
            Edge.gradient_edge(input)
            for input in args
        ]
        self.topological_nr = max([edge.input_nr for edge in self.next_edges if edge is not None] + [0]) + 1
        
        if self.ctx is None:
            self.ctx = Context()
    
        with no_grad():
            outputs = self.forward(self.ctx, *args, **kwargs)
        
        not_tuple = not isinstance(outputs, (list, tuple))
        if not_tuple:
            outputs = (outputs,)
        
        for i, output in enumerate(outputs):
            if not isinstance(output, Tensor):
                continue
            output.output_nr = i
            output.grad_fn = self
            output.requires_grad = True
        
        return outputs[0] if not_tuple else outputs
    
    def run(self, *args):
        with no_grad():
            grad_inputs = self.backward(self.ctx, *args)
        
        grad_inputs = wrap_tuple(grad_inputs)
        
        assert len(grad_inputs) == len(self.next_edges), "Number of gradients and inputs mismatch"
        
        return grad_inputs

class AccumulateGrad(Function):
    """
    Accumulate gradient to .grad field
    
    grad_fn for leaf tensors
    """
    def __init__(self, input: Tensor):
        super().__init__()
        self.ctx = Context()
        self.ctx.input = input # bypass tensor packing
    
    # this forward should never be called
    @staticmethod
    def forward(ctx: Context):
        return None
    
    @staticmethod
    def backward(ctx: Context, output_grad: Tensor):
        if ctx.input.requires_grad:
            if ctx.input.grad is None:
                ctx.input.grad = zeros_like(ctx.input)
            ctx.input.grad += output_grad
        return ()    

"""
    clone contiguous
"""

class Clone(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        print("CLONE:", input.shape)
        return input.clone()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output

class Contiguous(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.contiguous()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output
    
"""
    subscriptor
"""

class Subscriptor(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, index_or_slice: Union[int, slice, List[int], List[slice]]):
        ctx.input_shape = input.shape
        ctx.index_or_slice = index_or_slice
        return input[index_or_slice]
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_input = zeros(ctx.input_shape)
        grad_input[ctx.index_or_slice].copy_(grad_output)
        return grad_input, None
    
"""
    Element-wise Binary and Unary Operators
"""

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return -input
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return -grad_output


def reduce_broadcast(grad_output: Tensor, input_shape: List[int], output_shape: List[int], end_dim: int = 0) -> Tensor: 
    input_shape = list(input_shape)
    output_shape = list(output_shape)
    aligned_input_shape = [1] * (len(grad_output.shape) - len(input_shape)) + input_shape
        
    sum_dims = []
    for idx, (s1, s2) in enumerate(zip(aligned_input_shape, grad_output.shape)):
        if s1 != s2:
            sum_dims.append(idx)
    end_dim = end_dim if end_dim > 0 else end_dim + len(output_shape)
    sum_dims = [dim for dim in sum_dims if dim < end_dim]
    
    if sum_dims:
        grad_output = grad_output.sum(tuple(sum_dims), keepdims=False)
    
    grad_output = grad_output.reshape(input_shape)
    return grad_output

def binary_op_forward_wrapper(forward_impl):
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.input1_shape = input1.shape
        ctx.input2_shape = input2.shape
        
        output = forward_impl(ctx, input1, input2)
        
        return output
        
    return forward

def binary_op_backward_wrapper(backward_impl):
    def backward(ctx: Context, grad_output: Tensor):
        grad_input1, grad_input2 = backward_impl(ctx, grad_output)
        
        grad_input1 = reduce_broadcast(grad_input1, ctx.input1_shape, grad_output.shape)
        grad_input2 = reduce_broadcast(grad_input2, ctx.input2_shape, grad_output.shape)
        
        return grad_input1, grad_input2
    
    return backward


class Add(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 + input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output, grad_output
    
class Sub(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 - input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output, -grad_output
    
class Mul(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1 * input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        input1, input2 = ctx.get_saved_tensors()
        grad_input1 = grad_output * input2
        grad_input2 = grad_output * input1
        return grad_input1, grad_input2
    
class Div(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1 / input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        input1, input2 = ctx.get_saved_tensors()
        grad_input1 = grad_output / input2
        grad_input2 = -grad_output * (input1 / (input2 * input2))
        return grad_input1, grad_input2
    
class Sign(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.input_shape = input.shape
        return input.sign()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        # raise RuntimeError("Sign is not differentiable")
        return zeros(ctx.input_shape)
    
class Abs(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.abs()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * input.sign()
    
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.sin()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * input.cos()

class Cos(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.cos()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * (-input.sin())

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        output = input.tanh()
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        return grad_output * (1 - output * output)

class Clamp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, min_val: float, max_val: float):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return input.clamp(min_val, max_val)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input_tensor = ctx.get_saved_tensors()[0]
        min_val = ctx.min_val
        max_val = ctx.max_val
        mask = (input_tensor > min_val) * (input_tensor < max_val)
        print("CLAMP MASK:", mask)
        print("CLAMP GRAD_OUTPUT:", grad_output)
        return grad_output * mask, None, None

class Log(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.log()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input = ctx.get_saved_tensors()[0]
        return grad_output * (1 / input)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        output = input.exp()
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        return grad_output * output

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, exponent: float): 
        ctx.save_for_backward(input)
        ctx.exponent = exponent
        return input.pow(exponent)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input_tensor = ctx.get_saved_tensors()[0]
        exponent = ctx.exponent
        return grad_output * exponent * input_tensor.pow(exponent - 1), None
    
class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        output = input.sqrt()
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        return grad_output * (1 / (2 * output))
    
"""
    Matrix Multiplication
"""

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        
        output = input1.matmul(input2)
        
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input1_raw, input2_raw = ctx.get_saved_tensors()
        isvec1 = input1_raw.dim() == 1
        isvec2 = input2_raw.dim() == 1
        
        input1 = input1_raw.unsqueeze() if isvec1 else input1_raw
        input2 = input2_raw.unsqueeze().transpose() if isvec2 else input2_raw
        
        if isvec1 and isvec2:
            grad_output = grad_output.reshape([1, 1])
        elif isvec1: # grad_output may have been broadcasted, so specify -2 as dim is necessary
            grad_output = grad_output.unsqueeze(-2)
        elif isvec2: 
            grad_output = grad_output.unsqueeze(-2).transpose()
        
        input2_transposed = input2.transpose()
        grad_input1 = grad_output.matmul(input2_transposed)
        
        input1_transposed = input1.transpose()
        grad_input2 = input1_transposed.matmul(grad_output)
        
        if grad_input1.shape != input1.shape:
            grad_input1 = reduce_broadcast(grad_input1, input1.shape, list(grad_input1.shape), end_dim=-2)
        
        if grad_input2.shape != input2.shape:
            grad_input2 = reduce_broadcast(grad_input2, input2.shape, list(grad_input2.shape), end_dim=-2)
        
        if isvec1 == 1:
            grad_input1 = grad_input1.squeeze()
        if isvec2:
            grad_input2 = grad_input2.transpose().squeeze()
        
        return grad_input1, grad_input2

"""
    Reduction and Normalization Operations
"""

def reduce_forward_wrapper(forward_impl):
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdims = keepdims
        
        output = forward_impl(ctx, input, dim, keepdims)
        
        return output
    
    return forward

class Sum(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        return input.sum(dim, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        if not ctx.keepdims:
            grad_output = grad_output.unsqueeze(ctx.dim)
            
        grad_input = grad_output.broadcast_to(ctx.input_shape)
        
        return grad_input, None, None
    
class Max(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        max_values, max_indices = input.max(dim, keepdims=keepdims)
        ctx.save_for_backward(max_indices)
        return max_values, max_indices
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor, grad_indices: Tensor = None):
        max_indices = ctx.get_saved_tensors()[0]
        if ctx.keepdims:
            max_indices = max_indices.squeeze(ctx.dim)
            grad_output = grad_output.squeeze(ctx.dim)
            
        grad_input = zeros(ctx.input_shape)
        grad_input.scatter_(dim=ctx.dim, index=max_indices, src=grad_output)
        
        return grad_input, None, None
    
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int):
        ctx.dim = dim
        output = input.softmax(dim)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        output = ctx.get_saved_tensors()[0]
        grad_input = output * (grad_output - (grad_output * output).sum(ctx.dim, keepdims=True))
        return grad_input, None
    
"""
    Shape Manipulation
"""

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, perm: List[int]):
        ctx.input_shape = input.shape
        ctx.perm = perm
        return input.permute(perm)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        # Reverse the permutation for the gradient
        reverse_perm = [-1] * len(ctx.perm)
        for i, p in enumerate(ctx.perm):
            reverse_perm[p] = i
        reverse_perm = [p for p in reverse_perm if p >= 0]
        
        return grad_output.permute(reverse_perm), None
    
class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim0: int, dim1: int):
        ctx.input_shape = input.shape
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        return input.transpose(dim0, dim1)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.transpose(ctx.dim0, ctx.dim1), None, None

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.input_shape = input.shape
        ctx.output_shape = tuple(shape)
        return input.reshape(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.reshape(ctx.input_shape), None
    
class View(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.input_shape = input.shape
        ctx.output_shape = tuple(shape)
        return input.view(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.view(ctx.input_shape), None
    
class Narrow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, start: int, length: int):
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.start = start
        ctx.length = length
        return input.narrow(dim, start, length)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_input = zeros(ctx.input_shape)
        grad_input.narrow(ctx.dim, ctx.start, ctx.length).copy_(grad_output)
        return grad_input, None, None, None
    
class Chunk(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, chunks: int, dim: int = 0):
        ctx.dim = dim
        return input.chunk(chunks, dim)
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        grad_input = Tensor.cat(grad_outputs, dim=ctx.dim)
        return grad_input, None, None
    
class Split(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, split: Union[int, List[int]], dim: int = 0):
        ctx.dim = dim
        return input.split(split, dim)

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        grad_input = Tensor.cat(grad_outputs, dim=ctx.dim)
        
        return grad_input, None, None
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        # dim must be passed in kwargs
        ctx.dim = dim
        
        return Tensor.stack(inputs, dim=dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        dim = ctx.dim
        shape_dim = grad_output.shape[dim]
        grad_inputs = grad_output.chunk(shape_dim, dim=dim)
        grad_inputs = [
            t.squeeze(dim) for t in grad_inputs
        ]
        return grad_inputs
    
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        # dim must be passed in kwargs
        ctx.dim = dim
        ctx.split_section = [
            t.shape[dim] for t in inputs
        ]
        
        return Tensor.cat(inputs, dim)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        dim = ctx.dim
        split_section = ctx.split_section
        grad_inputs = grad_output.split(split_section, dim=dim)
        
        return grad_inputs
        
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        ctx.dim = dim
        return input.squeeze(dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):    
        return grad_output.unsqueeze(ctx.dim), None
    
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        ctx.dim = dim
        return input.unsqueeze(dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.squeeze(ctx.dim), None
    
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.input_shape = input.shape
        return input.broadcast_to(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input_shape = ctx.input_shape
        output_shape = grad_output.shape
        grad_input = reduce_broadcast(grad_output, input_shape, output_shape)

        return grad_input, None
    
class Broadcast(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor):
        ctx.input_shapes = [input.shape for input in inputs]
        
        return Tensor.broadcast(*inputs)
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        assert len(grad_outputs) == len(ctx.input_shapes), "Number of gradients must match number of inputs"
        
        grad_inputs = [
            reduce_broadcast(grad_output, input_shape, grad_output.shape) for grad_output, input_shape in zip(grad_outputs, ctx.input_shapes)
        ]
    
        return grad_inputs