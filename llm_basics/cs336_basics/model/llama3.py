import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLLM
from ..layer import Embedding, RMSNorm, Linear, TransformerBlock, RotaryPositionalEmbeddingPytorch, SwiGLU, scaled_dot_product_attention

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, apply_rotary_pos_emb

import logging

logger = logging.getLogger(__name__)
class LlamaAttentionWithHFRope(nn.Module):
    """Attention that uses HuggingFace RoPE directly"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, positional_encoder: LlamaRotaryEmbedding, attention_type: str = "eager"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = Linear(d_model, num_heads * self.head_dim)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.o_proj = Linear(num_heads * self.head_dim, d_model)
        
        # Store HF RoPE
        self.positional_encoder = positional_encoder
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None):
        bsz, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply HF RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        cos, sin = self.positional_encoder(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV heads
        if self.n_rep > 1:
            batch, num_kv_heads, slen, head_dim = k.shape
            k = k[:, :, None, :, :].expand(batch, num_kv_heads, self.n_rep, slen, head_dim)
            k = k.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)
            
            v = v[:, :, None, :, :].expand(batch, num_kv_heads, self.n_rep, slen, head_dim)
            v = v.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)
        
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_output = scaled_dot_product_attention(q, k, v, mask = mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output


class TransformerBlockWithHFRope(nn.Module):
    """Transformer block using HF RoPE"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, d_ff: int, 
                 positional_encoder: LlamaRotaryEmbedding, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.attn = LlamaAttentionWithHFRope(d_model, num_heads, num_kv_heads, positional_encoder)
        
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.ln2 = RMSNorm(d_model, eps=rms_norm_eps)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None):
        # Pre-norm transformer
        x_attn = self.attn(self.ln1(x), position_ids=position_ids)
        x = x + x_attn
        
        x_ffn = self.ffn(self.ln2(x))
        x = x + x_ffn
        
        return x


class Llama3(BaseLLM):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
    
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.positional_encoder = RotaryPositionalEmbeddingPytorch(
            context_length=config.context_length, 
            d_model=config.d_model // config.num_heads,  # Use head_dim instead of d_model
            theta=config.rope_theta
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    d_ff=config.d_ff,
                    positional_encoder=self.positional_encoder,
                    use_grouped_query_attention=True,
                    rms_norm_eps=config.rms_norm_eps,
                    layer_idx=idx,
                )
                for idx in range(config.num_layers)
            ]
        )
        self.ln_final = RMSNorm(config.d_model, eps = config.rms_norm_eps)

        self.lm_head = Linear(config.d_model, config.vocab_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.6f}M")
        logger.info(f"number of all parameters: {self.get_num_params(False) / 1e6:.6f}M")
        
    def forward(self, x):
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)
    

class Llama3DirectHFRope(nn.Module):
    """Llama3 model that directly uses HF RoPE"""
    
    def __init__(self, config, positional_encoder: LlamaRotaryEmbedding):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        
        # Layers with HF RoPE
        self.layers = nn.ModuleList([
            TransformerBlockWithHFRope(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                d_ff=config.d_ff,
                positional_encoder=positional_encoder,
                rms_norm_eps=config.rms_norm_eps
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm and output
        self.ln_final = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.d_model, config.vocab_size)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        self.num_layers = config.num_layers
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None):
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        
        # Embeddings
        h = self.token_embeddings(x)
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, position_ids=position_ids)
        
        # Final norm and output
        h = self.ln_final(h)
        logits = self.lm_head(h)
        
        return logits