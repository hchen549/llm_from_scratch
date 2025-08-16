import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLLM
from ..layer import Embedding, RMSNorm, Linear, TransformerBlock, RotaryPositionalEmbeddingPytorch

import logging

logger = logging.getLogger(__name__)

class Llama3(BaseLLM):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
    
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.positional_encoder = RotaryPositionalEmbeddingPytorch(context_length=config.context_length, d_model=config.d_model, theta=config.rope_theta)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    d_ff=config.d_ff,
                    positional_encoder=self.positional_encoder,
                    use_grouped_query_attention=True,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_final = RMSNorm(config.d_model, eps = config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = self.token_embeddings
        else:
            self.lm_head = Linear(config.d_model, config.vocab_size)

        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.6f}M")
        logger.info(f"number of all parameters: {self.get_num_params(False) / 1e6:.6f}M")
        
    def forward(self, x):
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)