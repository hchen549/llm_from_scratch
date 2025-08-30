"""Direct integration of HuggingFace RoPE into custom model for exact matching"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from config import model_config
from llm_basics.cs336_basics.model.llama3 import Llama3DirectHFRope, Llama3DirectHFRopeFastInference

from llm_basics.cs336_basics.layer import TransformerBlock, RMSNorm, Linear, Embedding, scaled_dot_product_attention, SwiGLU
import logging
from example.llama.load_hf import get_parameter_mapping, get_variable_mapping, test_model

logging.basicConfig(level=logging.INFO)


def load_model(hf_model_name, model_type = "llama3_hf_rope", attention_type = "paged"):
     # Load HF model with eager attention for exact matching
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # eager or flash_attention_2
    ).cuda()
    
    # Get HF RoPE
    hf_rope = hf_model.model.rotary_emb
    # Create custom config
    hf_config = hf_model.config
    my_model_cfg = model_config.LLamaConfig()
    
    # Update config from HF
    for key, value in get_variable_mapping(hf_to_custom=True).items():
        if hasattr(hf_config, key) and hasattr(my_model_cfg, value):
            setattr(my_model_cfg, value, getattr(hf_config, key))
    
    # Create custom model with HF RoPE
    if model_type == "llama3_hf_rope":
        model = Llama3DirectHFRope(my_model_cfg, hf_rope)
    elif model_type == "llama3_hf_rope_fast_inference":
        model = Llama3DirectHFRopeFastInference(my_model_cfg, hf_rope, attention_type=attention_type)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    model = model.to(torch.bfloat16).cuda()
    
    # Load weights
    layer_mapping = get_parameter_mapping(my_model_cfg, hf_to_custom=False)
    hf_state_dict = hf_model.state_dict()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            hf_param_name = layer_mapping[name]
            if hf_param_name in hf_state_dict:
                param.copy_(hf_state_dict[hf_param_name])
                logging.debug(f"Copied {hf_param_name} -> {name}")
    
    return hf_model, model

def load_and_test():
    # load hf_model and my model
    hf_model, model = load_model("meta-llama/Llama-3.2-1B-Instruct",)
    
    # Test the model
    test_results = test_model(hf_model, model)
    return hf_model, model, test_results


if __name__ == "__main__":
    hf_model, model, results = load_and_test()
