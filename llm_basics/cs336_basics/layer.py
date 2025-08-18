from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Bool, Int
from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFRMSNorm


from .nn_utils import softmax

logger = logging.getLogger(__name__)



class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            d_in: int
                The number of input features.
            d_out: int
                The number of output features.
        """
        
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std),
            requires_grad=True
        )
    
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]
    
    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        # NOTE: in practice, many implementations will
        # manually upcast the input to fp32 here to prevent overflow when you
        # square the input.
        # https://github.com/pytorch/pytorch/issues/66707
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        # Match HuggingFace's order: convert to dtype first, then multiply by weight
        return self.weight * x.to(in_dtype)
    
    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"


class RotaryPositionalEmbeddingPytorch(nn.Module):
    def __init__(
        self, 
        context_length: int, 
        d_model: int, 
        theta: float = 10000.0, 
        device = None,
        rope_type: str = "default",
        rope_scaling: dict = None
    ):
        super().__init__()
        self.theta = theta
        self.d_model = d_model
        self.context_length = context_length
        self.rope_type = rope_type
        self.rope_scaling = rope_scaling or {}

        assert d_model % 2 == 0
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Apply scaling if using Llama3 RoPE
        if self.rope_type == "llama3" and self.rope_scaling:
            inv_freq = self._apply_llama3_scaling(inv_freq)
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache for all positions
        self._build_cache(context_length, device)
    
    def _apply_llama3_scaling(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """Apply Llama3 RoPE scaling to inverse frequencies."""
        import math
        import numpy as np
        
        scaling_factor = self.rope_scaling.get('factor', 32.0)
        low_freq_factor = self.rope_scaling.get('low_freq_factor', 1.0)
        high_freq_factor = self.rope_scaling.get('high_freq_factor', 4.0)
        original_max_pos = self.rope_scaling.get('original_max_position_embeddings', 8192)
        
        # Calculate wavelengths
        wavelengths = 2 * math.pi / inv_freq
        
        # Define thresholds
        low_wavelen_threshold = original_max_pos * 2
        high_wavelen_threshold = original_max_pos / (2 * high_freq_factor)
        
        # Apply scaling based on wavelength
        ratio = torch.ones_like(inv_freq)
        
        for i, wavelen in enumerate(wavelengths):
            if wavelen < high_wavelen_threshold:
                # High frequency: no scaling
                ratio[i] = 1.0
            elif wavelen > low_wavelen_threshold:
                # Low frequency: scale by 1/factor
                ratio[i] = 1.0 / scaling_factor
            else:
                # Smooth interpolation
                t = (math.log(wavelen) - math.log(high_wavelen_threshold)) / \
                    (math.log(low_wavelen_threshold) - math.log(high_wavelen_threshold))
                t = np.clip(t, 0, 1)
                # Smooth step: 3t^2 - 2t^3
                smooth_t = t * t * (3 - 2 * t)
                ratio[i] = 1.0 * (1 - smooth_t) + (1.0 / scaling_factor) * smooth_t
        
        return inv_freq * ratio

    def _build_cache(self, seq_len, device):
        # Build cos/sin cache for all positions
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # Create cos and sin embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x shape: [batch, heads, seq, head_dim] for attention
        # token_positions shape: [batch, seq] or just [seq]
        
        # Get sequence positions
        if token_positions.dim() == 1:
            # If 1D, it's a sequence of positions
            seq_positions = token_positions
        else:
            # If 2D [batch, seq], take first batch since positions should be same across batch
            seq_positions = token_positions[0] if token_positions.shape[0] > 0 else token_positions.flatten()
        
        # Get cos and sin for the positions
        cos = self.cos_cached[seq_positions]  # [seq, head_dim]
        sin = self.sin_cached[seq_positions]  # [seq, head_dim]
        
        # Add dimensions for proper broadcasting with x shape [batch, heads, seq, head_dim]
        # We need to unsqueeze to get [1, 1, seq, head_dim]
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Apply rotary embedding using HuggingFace's approach
        x_rotated = (x * cos) + (self.rotate_half(x) * sin)
        
        return x_rotated


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        positional_encoder: RotaryEmbedding
            The RoPE module to use.
        layer_idx: int
            The index of this layer in the model (required for HF attention).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding | RotaryPositionalEmbeddingPytorch,
        use_grouped_query_attention: bool = False,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()

        if use_grouped_query_attention:
            self.attn = LlamaAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                positional_encoder=positional_encoder,
            )
        else:
            self.attn = CausalMultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                positional_encoder=positional_encoder,
            )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.ln2 = RMSNorm(d_model, eps=rms_norm_eps)
        self.use_grouped_query_attention = use_grouped_query_attention


    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, hf_rope_cos_sin: tuple = None):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.
            position_ids: Optional tensor of shape `(batch_size, sequence_length)`.
                Position IDs for the input sequence.
            attention_mask: Optional tensor of shape `(batch_size, sequence_length)`.
                Attention mask for the input sequence.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        # Both custom attention implementations return a single tensor
        if self.use_grouped_query_attention and hasattr(self.attn, 'use_hf_rope') and self.attn.use_hf_rope:
            # Pass HF RoPE cos/sin to attention
            x_attn = self.attn(self.ln1(x), token_positions=position_ids, hf_rope_cos_sin=hf_rope_cos_sin)
        elif position_ids is not None:
            x_attn = self.attn(self.ln1(x), token_positions=position_ids)
        else:
            x_attn = self.attn(self.ln1(x))
        
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    attention_weights = softmax(attention_scores, dim=-1, dtype = torch.float32).to(Q.dtype)

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        positional_encoder: RotaryEmbedding
            The RoPE module to use.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.o_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )  # fmt: skip

        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

        # Duplicate token positions for each head
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)

        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output

def silu(x: torch.Tensor):
    # Use PyTorch's F.silu for exact numerical compatibility with HuggingFace
    return F.silu(x)


class LlamaAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, positional_encoder: RotaryEmbedding | RotaryPositionalEmbeddingPytorch | None = None, use_hf_rope: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = d_model // num_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.n_rep = num_heads // num_kv_heads

        self.q_proj = Linear(d_model, num_heads * self.head_dim)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.output_proj = Linear(num_heads * self.head_dim, d_model)

        self.positional_encoder = positional_encoder
        self.use_hf_rope = use_hf_rope
        
        # If using HF RoPE, we'll get cos/sin from outside
        self._hf_cos = None
        self._hf_sin = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, hf_rope_cos_sin: tuple = None):
        
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x) # (b, seq, d_model)
        k = self.k_proj(x) # (b, seq, d_model/n_kv_heads)
        v = self.v_proj(x) # (b, seq, d_model/n_kv_heads)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (b, num_heads, seq, d)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # (b, num_kv_heads, seq, d)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # (b, num_kv_heads, seq, d)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # Apply RoPE BEFORE repeating KV heads (following HuggingFace implementation)
        if self.use_hf_rope and hf_rope_cos_sin is not None:
            # Use HuggingFace RoPE directly
            cos, sin = hf_rope_cos_sin
            # Import HF's apply_rotary_pos_emb
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            # Use custom RoPE
            q = self.positional_encoder(q, token_positions = token_positions )
            k = self.positional_encoder(k, token_positions = token_positions )

        # Repeat KV heads AFTER applying RoPE using HuggingFace's method
        # This is more efficient than repeat_interleave and matches HF's implementation
        if self.n_rep > 1:
            batch, num_kv_heads, slen, head_dim = k.shape
            k = k[:, :, None, :, :].expand(batch, num_kv_heads, self.n_rep, slen, head_dim)
            k = k.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)
            
            v = v[:, :, None, :, :].expand(batch, num_kv_heads, self.n_rep, slen, head_dim)
            v = v.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)

        mask = 1 - torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.bool()
        
        attention = scaled_dot_product_attention(q, k, v, mask)
        
        # Transpose back and reshape
        attention = attention.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()

        output = self.output_proj(attention)

        return output
        
        
