from typing import Optional
from ..tensor import Tensor
from .module import Module, Parameter
from . import init
import math


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor.empty(out_features))
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
        output = x @ self.weight.T
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
    
    def __init__(self, normalized_shape, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.normailzed_numel = 1
        for size_dim in self.normalized_shape:
            self.normailzed_numel *= size_dim
            
        self.eps = eps
        
        if affine:
            self.affine = Linear(self.normailzed_numel, self.normailzed_numel, bias=True)
        else:
            self.affine = None

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        x = x.reshape((-1, self.normailzed_numel))
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x = self.affine(x)
        return x.reshape(input_shape)
    
    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, affine={self.affine is not None}"

class BatchNorm(Module):
    pass

class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None):
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = scores.softmax(dim=-1)

        context = attn_weights @ v

        context = context.transpose(1, 2).reshape(batch_size, seq_len_q, self.embed_dim)
        
        return self.out_proj(context)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, bias={self.q_proj.bias is not None}"