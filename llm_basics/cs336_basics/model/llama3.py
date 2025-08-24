import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .base import BaseLLM
from ..layer import Embedding, RMSNorm, Linear, TransformerBlock, RotaryPositionalEmbeddingPytorch, SwiGLU, scaled_dot_product_attention

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import logging


@dataclass
class ModelOutput:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

logger = logging.getLogger(__name__)
class LlamaAttentionWithHFRope(nn.Module):
    """Attention that uses HuggingFace RoPE directly"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, positional_encoder: LlamaRotaryEmbedding, context_length: int = 1024, attention_type: str = "eager"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.context_length = context_length
        
        # Projections
        self.q_proj = Linear(d_model, num_heads * self.head_dim)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.o_proj = Linear(num_heads * self.head_dim, d_model)
        
        # Store HF RoPE
        self.positional_encoder = positional_encoder
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
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
            k_repeated = k[:, :, None, :, :].expand(batch, num_kv_heads, self.n_rep, slen, head_dim)
            k_repeated = k_repeated.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)
            
            v_repeated = v[:, :, None, :, :].expand(batch, num_kv_heads, self.n_rep, slen, head_dim)
            v_repeated = v_repeated.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)
        else:
            k_repeated = k
            v_repeated = v
        
        if mask is None:
            mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()   

        attn_output = scaled_dot_product_attention(q, k_repeated, v_repeated, mask = mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output, (k, v)


def fast_attention_forward(self, x: torch.Tensor, position_ids: torch.Tensor = None, past_key_values: Tuple[torch.Tensor, torch.Tensor] = None, is_first_decode: bool = False, mask: Optional[torch.Tensor] = None):
    bsz, curr_seq_len, _ = x.shape

    assert past_key_values is not None, "past_key_values is required for fast inference"
    past_kv_seq_len = 0 if past_key_values is None else past_key_values[0].shape[2]
    new_kv_seq_len = past_kv_seq_len + curr_seq_len
    
    if past_key_values is not None and is_first_decode:
        past_key, past_val = past_key_values
        self.kv_cache = torch.empty((2, bsz, self.context_length, self.num_kv_heads, self.head_dim), device = x.device, dtype = x.dtype)
        self.kv_cache[0][:, :past_kv_seq_len] = past_key.transpose(1, 2)
        self.kv_cache[1][:, :past_kv_seq_len] = past_val.transpose(1, 2)

    q = self.q_proj(x).view(bsz, curr_seq_len, self.num_heads, -1).transpose(1, 2) # (bsz, num_heads, curr_seq_len, head_dim)
    k = self.k_proj(x).view(bsz, curr_seq_len, self.num_kv_heads, -1).transpose(1, 2) # (bsz, num_kv_heads, curr_seq_len, head_dim)
    v = self.v_proj(x).view(bsz, curr_seq_len, self.num_kv_heads, -1).transpose(1, 2) # (bsz, num_kv_heads, curr_seq_len, head_dim)

    # apply positional_embedding to q, k
    if position_ids is None:
        position_ids = torch.arange(curr_seq_len, device=x.device).unsqueeze(0)
    
    cos, sin = self.positional_encoder(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # save computed k and v to cache
    self.kv_cache[0][:, past_kv_seq_len:new_kv_seq_len] = k.transpose(1, 2)
    self.kv_cache[1][:, past_kv_seq_len:new_kv_seq_len] = v.transpose(1, 2)

   
    keys = self.kv_cache[0][:, :new_kv_seq_len] # (bsz, new_kv_seq_len, num_kv_heads, head_dim)
    values = self.kv_cache[1][:, :new_kv_seq_len] # (bsz, new_kv_seq_len, num_kv_heads, head_dim)
    
    keys = keys.transpose(1, 2) # (bsz, num_kv_heads, new_kv_seq_len, head_dim)
    values = values.transpose(1, 2) # (bsz, num_kv_heads, new_kv_seq_len, head_dim)


    # apply gqa to kv
    if self.n_rep > 1:
        k_repeated = keys.unsqueeze(2).expand(bsz, self.num_kv_heads, self.n_rep, new_kv_seq_len, self.head_dim).reshape(bsz, self.num_heads, new_kv_seq_len, self.head_dim)
        v_repeated = values.unsqueeze(2).expand(bsz, self.num_kv_heads, self.n_rep, new_kv_seq_len, self.head_dim).reshape(bsz, self.num_heads, new_kv_seq_len, self.head_dim)

    else:
        k_repeated = keys
        v_repeated = values

    # get the mask of the attention
    # if mask is not None:
    #     mask = 

    # apply attention
    attn_output = scaled_dot_product_attention(q, k_repeated, v_repeated, mask = mask) # (bsz, num_heads, curr_seq_len, head_dim)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, curr_seq_len, -1) # (bsz, curr_seq_len, d_model)
    return self.o_proj(attn_output), (keys, values)
    
def fast_llama_forward(self, x: torch.Tensor, position_ids: torch.Tensor = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, mask: Optional[torch.Tensor] = None):
    if position_ids is None:
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    
    assert past_key_values is not None, "past_key_values is required for fast inference"
    h = self.token_embeddings(x)
    
    new_kv_values = []
    for i, layer in enumerate(self.layers):
        layer_past_kv = past_key_values[i] 

        is_first_decode = not hasattr(layer.attn, 'kv_cache')

        h_attn, kv_value = fast_attention_forward(
            layer.attn, 
            layer.ln1(h), 
            position_ids=position_ids,
            past_key_values=layer_past_kv, 
            is_first_decode=is_first_decode,
            mask=mask
        )
        h = h + h_attn
        
        h_ffn = layer.ffn(layer.ln2(h))
        h = h + h_ffn
        new_kv_values.append(kv_value)
    
    h = self.ln_final(h)
    logits = self.lm_head(h)
    
    return ModelOutput(logits=logits, past_key_values=new_kv_values)



class TransformerBlockWithHFRope(nn.Module):
    """Transformer block using HF RoPE"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, d_ff: int, 
                 positional_encoder: LlamaRotaryEmbedding, rms_norm_eps: float = 1e-6, context_length: int = 1024):
        super().__init__()
        self.attn = LlamaAttentionWithHFRope(d_model, num_heads, num_kv_heads, positional_encoder, context_length=context_length)
        
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.ln2 = RMSNorm(d_model, eps=rms_norm_eps)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        # Pre-norm transformer
        x_attn, kv_value = self.attn(self.ln1(x), position_ids=position_ids)
        x = x + x_attn
        
        x_ffn = self.ffn(self.ln2(x))
        x = x + x_ffn
        
        return x, kv_value


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
                    context_length=config.context_length,
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
        
    def forward(self, x, position_ids=None, past_key_values=None, mask=None):
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)

 
    

class Llama3DirectHFRope(Llama3):
    """Llama3 model that directly uses HF RoPE"""
    
    def __init__(self, config, positional_encoder: LlamaRotaryEmbedding):
        super().__init__(config)
        
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

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None, past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None, mask: Optional[torch.Tensor] = None):
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        
        # Embeddings
        h = self.token_embeddings(x)
        
        kv_values = []
        # Transformer layers
        for layer in self.layers:
            h, kv_value = layer(h, position_ids=position_ids)
            kv_values.append(kv_value)
        
        # Final norm and output
        h = self.ln_final(h)
        logits = self.lm_head(h)
        
        return ModelOutput(logits=logits, past_key_values=kv_values)
    
    @torch.inference_mode()
    def generate(self, x, max_new_tokens, temperature = 1, top_p = 1.0, eos_token_id = None):
        past_key_values = None
        
        for i in range(min(max_new_tokens, self.config.context_length - x.shape[1])):
           
            input_ids, position_ids, mask = self.prepare_generation_inputs(x, past_key_values=past_key_values)
            output = self.forward(input_ids, position_ids=position_ids, past_key_values=past_key_values, mask=mask)
            
            logits = output.logits[:, -1, :] # (bsz, vocab_size)
            past_key_values = output.past_key_values

            scaled_logits = torch.softmax(logits/temperature, dim=-1) # (bsz, vocab_size)
            next_token_id = torch.argmax(scaled_logits, dim=-1, keepdim=True) # (bsz, 1)
            
            print(next_token_id)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            
            x = torch.cat((x, next_token_id), dim=-1)

        return x

    def prepare_generation_inputs(self, x, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool() 

        return x, position_ids, mask


    
class Llama3DirectHFRopeFastInference(Llama3DirectHFRope):
    """Llama3 model that directly uses HF RoPE"""
    
    def __init__(self, config, positional_encoder: LlamaRotaryEmbedding):
        super().__init__(config, positional_encoder)
        
        
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
        
    
    def forward(self, x: torch.Tensor, 
                position_ids: torch.Tensor = None, 
                past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None):

        if past_key_values is None:
            return super().forward(x, position_ids=position_ids, past_key_values=past_key_values, mask=mask)
        else:
            return fast_llama_forward(self, x=x, position_ids=position_ids, past_key_values=past_key_values, mask=mask)
    
    @torch.inference_mode()
    def generate(self, x, max_new_tokens, temperature = 1, top_p = 1.0, eos_token_id = None):
        past_key_values = None
        
        for i in range(min(max_new_tokens, self.config.context_length - x.shape[1])):
           
            input_ids, position_ids, mask = self.prepare_generation_inputs(x, past_key_values=past_key_values)
            output = self.forward(input_ids, position_ids=position_ids, past_key_values=past_key_values, mask=mask)
            
            logits = output.logits[:, -1, :] # (bsz, vocab_size)
            past_key_values = output.past_key_values

            scaled_logits = torch.softmax(logits/temperature, dim=-1) # (bsz, vocab_size)
            next_token_id = torch.argmax(scaled_logits, dim=-1, keepdim=True) # (bsz, 1)
            
            print("generated token i: ", next_token_id)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            
            x = torch.cat((x, next_token_id), dim=-1)

        return x

    
    @torch.inference_mode()
    def prepare_generation_inputs(self, x, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        seq_len = x.shape[1] # (bsz, seq_len)
        
        if past_key_values is None:
            # prefill
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            return x, position_ids, mask
        else:
            # decoding
            past_kv_len = past_key_values[0][0].shape[-2]
            position_ids = torch.tensor([past_kv_len + seq_len -1], device=x.device).unsqueeze(0)
            mask = None
            return x[:, -1:], position_ids, mask