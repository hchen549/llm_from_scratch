import os
import torch
import torch.nn as nn
from ..layer import Embedding, RotaryEmbedding, RMSNorm, Linear, TransformerBlock  
from jaxtyping import Float, Int
from torch import Tensor
import torch.nn.functional as F
import json

import logging

logger = logging.getLogger(__name__)

class BaseLLM(nn.Module):

    def __init__(self):
        super().__init__()

    def get_num_params(self, non_embedding=True):
        param_counts = 0
        for name, param in self.named_parameters():
            param_counts += param.numel()

        if non_embedding:
            param_counts -= self.lm_head.weight.numel()

        return param_counts


class BasicsTransformerLM(BaseLLM):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta: float
            The theta value for the RoPE positional encoding.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length,
            dim=d_head,
            theta=rope_theta
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=rms_norm_eps)
        self.lm_head = Linear(d_model, vocab_size)

        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.6f}M")
        logger.info(f"number of all parameters: {self.get_num_params(False) / 1e6:.6f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
            distribution for each token.
        """
        _, sequence_length = x.size()

        # (batch size, sequence_length, d_model)
        x = self.token_embeddings(x)

        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)

        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)

        # (batch size, sequence_length, vocab_size)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = F.softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
