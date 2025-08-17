
from dataclasses import fields, asdict
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import model_config
from llm_basics.cs336_basics.model.llama3 import Llama3
import logging

logging.basicConfig(level=logging.INFO)
import torch
from torch.distributed.tensor import DTensor, distribute_tensor


def get_parameter_mapping(model_config, hf_to_custom):
    hf_to_custom_layer_mapping = {}
    hf_to_custom_layer_mapping["model.embed_tokens.weight"] = "token_embeddings.weight"
    for layer_i in range(model_config.num_layers):
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = f"layers.{layer_i}.attn.q_proj.weight"
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = f"layers.{layer_i}.attn.k_proj.weight"
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = f"layers.{layer_i}.attn.v_proj.weight"
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = f"layers.{layer_i}.attn.output_proj.weight"

        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = f"layers.{layer_i}.ffn.w1.weight"
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.mlp.up_proj.weight"] = f"layers.{layer_i}.ffn.w3.weight"
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.mlp.down_proj.weight"] = f"layers.{layer_i}.ffn.w2.weight"

        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.input_layernorm.weight"] = f"layers.{layer_i}.ln1.weight"
        hf_to_custom_layer_mapping[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = f"layers.{layer_i}.ln2.weight"

    hf_to_custom_layer_mapping["model.norm.weight"] = "ln_final.weight"
    hf_to_custom_layer_mapping["lm_head.weight"] = "lm_head.weight"

    if hf_to_custom:
        return hf_to_custom_layer_mapping
    else:
        return {val: key for key, val in hf_to_custom_layer_mapping.items()}


def get_variable_mapping(hf_to_custom = True):
    hf_to_custom_model_mapping = {
        "vocab_size": "vocab_size",
        "max_position_embeddings": "context_length",
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "d_ff",
        "rope_theta": "rope_theta",
        "rms_norm_eps": "rms_norm_eps",
        "tie_word_embeddings": "tie_word_embeddings"
    }

    if hf_to_custom:
        return hf_to_custom_model_mapping
    else:
        return {value: key for key, value in hf_to_custom_model_mapping.items()}

def load_hf_model():
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", 
        torch_dtype=torch.bfloat16,
        device_map = "cuda:0"
    )
    hf_model_cfg = hf_model.config


    my_model_cfg = model_config.LLamaConfig()
    for key, value in get_variable_mapping(hf_to_custom = True).items():
        if hasattr(hf_model_cfg, key) and hasattr(my_model_cfg, value):
            setattr(my_model_cfg, value, getattr(hf_model_cfg, key))
        else:
            logging.warning(f"Field {key} not found in hf_model_cfg or field {value} not found in my_model_cfg")

    logging.info(f"my_model_cfg: {my_model_cfg}")
    logging.info(f"hf_model_cfg: {hf_model_cfg}")

    model = Llama3(my_model_cfg)

    layer_mapping = get_parameter_mapping(my_model_cfg, hf_to_custom=False)

    hf_state_dict = hf_model.state_dict()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            hf_param_name = layer_mapping[name]
            if hf_param_name in hf_state_dict:
                source_tensor = hf_state_dict[hf_param_name]
                param.copy_(source_tensor)
                    
                logging.info(f"Copied {hf_param_name} -> {name}")
            else:
                logging.warning(f"HF parameter {hf_param_name} not found in state_dict")

    return hf_model, model
        

def test_model(hf_model, model):
    hf_model.to("cuda:0")
    model.to("cuda:0")

    hf_model.eval()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    

if __name__ == "__main__":
    hf_model, model = load_hf_model()
  
    
    time.sleep(1000000)


