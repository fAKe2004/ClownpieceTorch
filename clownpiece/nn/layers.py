# Linear, Embedding, LayerNorm, BatchNorm, MultiheadAttention

from typing import Optional
from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module, Parameter, Buffer
from . import init
import math


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(Tensor.empty((out_features,)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(1 / self.in_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
          init.uniform_(self.bias, -bound, bound)
        # or equvialently, use 
        # init.kaiming_uniform_(self.weight, a = math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.transpose()
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class Embedding(Module):
    def __init__(self, num_embd: int, embd_dim: int):
        super().__init__()
        self.num_embd = num_embd
        self.embd_dim = embd_dim
        self.weight = Parameter(Tensor.randn((num_embd, embd_dim)))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]

    def extra_repr(self):
        return f"num_embd={self.num_embd}, embd_dim={self.embd_dim}"

class LayerNorm(Module):
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
            
        self.eps = eps
        
        if affine:
            self.affine = Linear(self.num_features, self.num_features, bias=True)
        else:
            self.affine = None
            self.register_buffer('affine.weight', None)
            self.register_buffer('affine.bias', None)

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        x = x.reshape([-1, self.num_features])
        mean = x.mean(dim=-1, keepdims=True)
        var = x.var(dim=-1, keepdims=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x = self.affine(x)
        return x.reshape(input_shape)
    
    def extra_repr(self):
        return f"num_features={self.num_features}, eps={self.eps}, affine={self.affine is not None}"

class BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        if affine:
            self.affine = Linear(num_features, num_features, bias=True)
        else:
            self.affine = None
            self.register_buffer('affine.weight', None)
            self.register_buffer('affine.bias', None)
        self.running_mean = Buffer(Tensor.zeros([num_features]))
        self.running_var = Buffer(Tensor.ones([num_features]))
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x_hat = self.affine(x_hat)
        return x_hat

    def extra_repr(self):
        return f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"

class MultiheadAttention(Module):
    def __init__(self, hidden_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_proj = Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None):

        batch_seq_shape = hidden_states.shape[:-1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        qkv_shape = batch_seq_shape + [self.num_heads, self.head_dim]

        q = q.reshape(
            qkv_shape
            ).transpose(-3, -2)
        k = k.reshape(
            qkv_shape
            ).transpose(-3, -2)
        v = v.reshape(
            qkv_shape
            ).transpose(-3, -2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + (1 - attn_mask) * 1e-6

        attn_weights = scores.softmax(dim=-1)

        context = attn_weights @ v

        context = context.transpose(1, 2).reshape(batch_seq_shape + [self.hidden_dim])
        
        return self.out_proj(context)

    def extra_repr(self):
        return f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, bias={self.q_proj.bias is not None}"