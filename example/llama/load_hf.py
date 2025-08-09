
from dataclasses import fields, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import model_config
from llm_basics.cs336_basics.model import BasicsTransformerLM
import logging


def get_variable_mapping(hf_to_custom = True):
    hf_to_custom_model_mapping = {
        "vocab_size": "vocab_size",
        "max_position_embeddings": "context_length",
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_heads",
        "intermediate_size": "d_ff",
        "rope_theta": "rope_theta",
        "rms_norm_eps": "rms_norm_eps"
    }

    if hf_to_custom_model_mapping:
        return hf_to_custom_model_mapping
    else:
        return {value: key for key, value in hf_to_custom_model_mapping.items()}

def load_hf_model():
    hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    hf_model_cfg = hf_model.config

    print("hf_model_cfg: ", hf_model_cfg)

    my_model_cfg = model_config.LLamaConfig()
    for key, value in get_variable_mapping(hf_to_custom = True).items():
        if hasattr(hf_model_cfg, key) and hasattr(my_model_cfg, value):
            setattr(my_model_cfg, value, getattr(hf_model_cfg, key))
        else:
            logging.warning(f"Field {key} not found in hf_model_cfg or field {value} not found in my_model_cfg")

    model = BasicsTransformerLM(**asdict(my_model_cfg))

    print(f"huggingface model: {hf_model}")
    print(f"my model: {model}")
    return model


if __name__ == "__main__":
    load_hf_model()


