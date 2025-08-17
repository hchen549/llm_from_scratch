from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLamaConfig:
    vocab_size: int = 32000
    context_length: int = 1024,
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12
    d_model: int = 768,
    d_ff: int = 3072,
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True