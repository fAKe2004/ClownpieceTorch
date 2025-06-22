from typing import Optional
from ..tensor import Tensor
from .module import Module, Parameter
import math


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        k = 1.0 / self.in_features
        limit = math.sqrt(k)
        self.weight = Parameter(Tensor.rand(out_features, in_features) * 2 * limit - limit)
        
        if bias:
            self.bias = Parameter(Tensor.rand(out_features) * 2 * limit - limit)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor.randn(num_embeddings, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]

    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(Tensor.ones(self.normalized_shape))
        self.bias = Parameter(Tensor.zeros(self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean = x.mean(axis=dims, keepdims=True)
        var = x.var(axis=dims, keepdims=True, unbiased=True)
        
        x_normalized = (x - mean) / ((var + self.eps).sqrt())
        
        return self.weight * x_normalized + self.bias

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"

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

    def __repr__(self):
        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"