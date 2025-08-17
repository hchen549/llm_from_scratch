import torch
import torch.nn as nn
from dataclasses import fields, asdict
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import model_config
from llm_basics.cs336_basics.model.llama3 import Llama3
import logging

logging.basicConfig(level=logging.INFO)
import torch
from torch.distributed.tensor import DTensor, distribute_tensor
import numpy as np
from typing import Dict, List, Tuple


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

    logging.info(f"HF Model Config Details:")
    logging.info(f"  vocab_size: {getattr(hf_model_cfg, 'vocab_size', 'N/A')}")
    logging.info(f"  max_position_embeddings: {getattr(hf_model_cfg, 'max_position_embeddings', 'N/A')}")
    logging.info(f"  hidden_size: {getattr(hf_model_cfg, 'hidden_size', 'N/A')}")
    logging.info(f"  num_hidden_layers: {getattr(hf_model_cfg, 'num_hidden_layers', 'N/A')}")
    logging.info(f"  num_attention_heads: {getattr(hf_model_cfg, 'num_attention_heads', 'N/A')}")
    logging.info(f"  num_key_value_heads: {getattr(hf_model_cfg, 'num_key_value_heads', 'N/A')}")
    logging.info(f"  intermediate_size: {getattr(hf_model_cfg, 'intermediate_size', 'N/A')}")
    logging.info(f"  rope_theta: {getattr(hf_model_cfg, 'rope_theta', 'N/A')}")

    my_model_cfg = model_config.LLamaConfig()
    logging.info(f"Initial Custom Model Config:")
    logging.info(f"  vocab_size: {my_model_cfg.vocab_size}")
    logging.info(f"  context_length: {my_model_cfg.context_length}")
    logging.info(f"  d_model: {my_model_cfg.d_model}")
    logging.info(f"  num_layers: {my_model_cfg.num_layers}")
    logging.info(f"  num_heads: {my_model_cfg.num_heads}")
    logging.info(f"  num_kv_heads: {my_model_cfg.num_kv_heads}")
    logging.info(f"  d_ff: {my_model_cfg.d_ff}")
    logging.info(f"  rope_theta: {my_model_cfg.rope_theta}")

    for key, value in get_variable_mapping(hf_to_custom = True).items():
        if hasattr(hf_model_cfg, key) and hasattr(my_model_cfg, value):
            old_val = getattr(my_model_cfg, value)
            new_val = getattr(hf_model_cfg, key)
            setattr(my_model_cfg, value, new_val)
            logging.info(f"Updated {value}: {old_val} -> {new_val}")
        else:
            logging.warning(f"Field {key} not found in hf_model_cfg or field {value} not found in my_model_cfg")

    logging.info(f"Final Custom Model Config:")
    logging.info(f"  vocab_size: {my_model_cfg.vocab_size}")
    logging.info(f"  context_length: {my_model_cfg.context_length}")
    logging.info(f"  d_model: {my_model_cfg.d_model}")
    logging.info(f"  num_layers: {my_model_cfg.num_layers}")
    logging.info(f"  num_heads: {my_model_cfg.num_heads}")
    logging.info(f"  num_kv_heads: {my_model_cfg.num_kv_heads}")
    logging.info(f"  d_ff: {my_model_cfg.d_ff}")
    logging.info(f"  rope_theta: {my_model_cfg.rope_theta}")
    logging.info(f"  rms_norm_eps: {my_model_cfg.rms_norm_eps}")
    logging.info(f"  tie_word_embeddings: {my_model_cfg.tie_word_embeddings}")

    model = Llama3(my_model_cfg)
    print(f"custom model dtype: {model.layers[0].ln1.weight.dtype}")

    model.to(torch.bfloat16)
    print(f"custom model dtype: {model.layers[0].ln1.weight.dtype}")

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


class HiddenStateExtractor:
    """Class to extract hidden states from models during forward pass"""
    
    def __init__(self):
        self.hidden_states = {}
        self.hooks = []
    
    def clear(self):
        """Clear stored hidden states"""
        self.hidden_states = {}
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def register_hf_hooks(self, model):
        """Register hooks for HuggingFace model with fine-grained layer outputs"""
        self.clear()
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # For transformer layers, take the first output (hidden states)
                    self.hidden_states[name] = output[0].detach().cpu()
                else:
                    self.hidden_states[name] = output.detach().cpu()
            return hook
        
        def make_input_hook(name):
            def hook(module, input):
                # Capture the input to the module
                if isinstance(input, tuple) and len(input) > 0:
                    self.hidden_states[name] = input[0].detach().cpu()
                else:
                    self.hidden_states[name] = input.detach().cpu()
            return hook
        
        # Hook embedding layer
        hook = model.model.embed_tokens.register_forward_hook(
            make_hook("embed_tokens")
        )
        self.hooks.append(hook)
        
        # Hook each transformer layer with fine-grained outputs
        for i, layer in enumerate(model.model.layers):
            # Capture layer input
            hook = layer.register_forward_pre_hook(
                make_input_hook(f"layer_{i}_input")
            )
            self.hooks.append(hook)
            
            # Capture input_layernorm output (equivalent to ln1)
            hook = layer.input_layernorm.register_forward_hook(
                make_hook(f"layer_{i}_ln1")
            )
            self.hooks.append(hook)
            
            # Capture self-attention output
            hook = layer.self_attn.register_forward_hook(
                make_hook(f"layer_{i}_attn")
            )
            self.hooks.append(hook)
            
            # Capture post_attention_layernorm output (equivalent to ln2)
            hook = layer.post_attention_layernorm.register_forward_hook(
                make_hook(f"layer_{i}_ln2")
            )
            self.hooks.append(hook)
            
            # Capture MLP output (equivalent to FFN)
            hook = layer.mlp.register_forward_hook(
                make_hook(f"layer_{i}_ffn")
            )
            self.hooks.append(hook)
            
            # Capture final layer output
            hook = layer.register_forward_hook(
                make_hook(f"layer_{i}_output")
            )
            self.hooks.append(hook)
        
        # Hook final layer norm
        hook = model.model.norm.register_forward_hook(
            make_hook("final_norm")
        )
        self.hooks.append(hook)
        
        # Hook lm_head
        hook = model.lm_head.register_forward_hook(
            make_hook("lm_head")
        )
        self.hooks.append(hook)
    
    def register_custom_hooks(self, model):
        """Register hooks for custom model with fine-grained layer outputs"""
        self.clear()
        
        def make_hook(name):
            def hook(module, input, output):
                self.hidden_states[name] = output.detach().cpu()
            return hook
        
        def make_input_hook(name):
            def hook(module, input):
                # Capture the input to the module
                if isinstance(input, tuple) and len(input) > 0:
                    self.hidden_states[name] = input[0].detach().cpu()
                else:
                    self.hidden_states[name] = input.detach().cpu()
            return hook
        
        # Hook embedding layer
        hook = model.token_embeddings.register_forward_hook(
            make_hook("embed_tokens")
        )
        self.hooks.append(hook)
        
        # Hook each transformer layer with fine-grained outputs
        for i, layer in enumerate(model.layers):
            # Capture layer input
            hook = layer.register_forward_pre_hook(
                make_input_hook(f"layer_{i}_input")
            )
            self.hooks.append(hook)
            
            # Capture ln1 output
            hook = layer.ln1.register_forward_hook(
                make_hook(f"layer_{i}_ln1")
            )
            self.hooks.append(hook)
            
            # Capture attention output (before residual)
            hook = layer.attn.register_forward_hook(
                make_hook(f"layer_{i}_attn")
            )
            self.hooks.append(hook)
            
            # Capture ln2 output
            hook = layer.ln2.register_forward_hook(
                make_hook(f"layer_{i}_ln2")
            )
            self.hooks.append(hook)
            
            # Capture FFN output (before residual)
            hook = layer.ffn.register_forward_hook(
                make_hook(f"layer_{i}_ffn")
            )
            self.hooks.append(hook)
            
            # Capture final layer output
            hook = layer.register_forward_hook(
                make_hook(f"layer_{i}_output")
            )
            self.hooks.append(hook)
        
        # Hook final layer norm
        hook = model.ln_final.register_forward_hook(
            make_hook("final_norm")
        )
        self.hooks.append(hook)
        
        # Hook lm_head
        hook = model.lm_head.register_forward_hook(
            make_hook("lm_head")
        )
        self.hooks.append(hook)


def compare_hidden_states(hf_states: Dict, custom_states: Dict, rtol: float = 1e-3, atol: float = 1e-5) -> Dict:
    """
    Compare hidden states between HF and custom models
    
    Args:
        hf_states: Hidden states from HuggingFace model
        custom_states: Hidden states from custom model  
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for layer_name in hf_states.keys():
        if layer_name not in custom_states:
            results[layer_name] = {
                "status": "MISSING",
                "error": f"Layer {layer_name} not found in custom model states"
            }
            continue
        
        hf_tensor = hf_states[layer_name]
        custom_tensor = custom_states[layer_name]
        
        # Convert to same dtype and device for comparison
        hf_tensor = hf_tensor.float()
        custom_tensor = custom_tensor.float()
        
        # Check shapes match
        if hf_tensor.shape != custom_tensor.shape:
            results[layer_name] = {
                "status": "SHAPE_MISMATCH",
                "hf_shape": hf_tensor.shape,
                "custom_shape": custom_tensor.shape,
                "error": f"Shape mismatch: HF {hf_tensor.shape} vs Custom {custom_tensor.shape}"
            }
            continue
        
        # Check if tensors are close
        are_close = torch.allclose(hf_tensor, custom_tensor, rtol=rtol, atol=atol)
        
        # Compute statistics
        diff = (hf_tensor - custom_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_diff = (diff / (hf_tensor.abs() + 1e-8)).mean().item()
        
        results[layer_name] = {
            "status": "PASS" if are_close else "FAIL",
            "max_absolute_diff": max_diff,
            "mean_absolute_diff": mean_diff,
            "mean_relative_diff": relative_diff,
            "shape": hf_tensor.shape,
            "all_close": are_close
        }
    
    return results


def test_model(hf_model, model):
    """Test that hidden states match between HF and custom models"""
    
    # Move models to same device and set to eval mode
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hf_model.to(device)
    model.to(device)
    hf_model.eval()
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test inputs
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        # "Hello world, this is a test sentence for comparison.",
        # "Artificial intelligence is transforming the way we work and live."
    ]
    
    all_results = {}
    
    for i, text in enumerate(test_texts):
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing with input {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        logging.info(f"{'='*60}")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        logging.info(f"Input shape: {input_ids.shape}")
        logging.info(f"Input tokens: {input_ids[0][:10].tolist()}...")
        
        # Create extractors
        hf_extractor = HiddenStateExtractor()
        custom_extractor = HiddenStateExtractor()
        
        try:
            # Extract hidden states from HF model
            hf_extractor.register_hf_hooks(hf_model)
            with torch.no_grad():
                hf_output = hf_model(input_ids)
            hf_states = hf_extractor.hidden_states.copy()
            hf_extractor.remove_hooks()
            
            # Extract hidden states from custom model  
            custom_extractor.register_custom_hooks(model)
            with torch.no_grad():
                custom_output = model(input_ids)
            custom_states = custom_extractor.hidden_states.copy()
            custom_extractor.remove_hooks()
            
            # Compare final outputs
            hf_logits = hf_output.logits.detach().cpu().float()
            custom_logits = custom_output.detach().cpu().float()
            
            final_output_close = torch.allclose(hf_logits, custom_logits, rtol=1e-4, atol=1e-6)
            max_output_diff = (hf_logits - custom_logits).abs().max().item()
            
            logging.info(f"Final output shapes - HF: {hf_logits.shape}, Custom: {custom_logits.shape}")
            logging.info(f"Final outputs match: {final_output_close}")
            logging.info(f"Max output difference: {max_output_diff:.2e}")
            
            # Compare hidden states layer by layer
            comparison_results = compare_hidden_states(hf_states, custom_states)
            
            # Print detailed results
            logging.info(f"\nHidden States Comparison Results:")
            
            # First, print embedding and final layers
            logging.info(f"\n{'='*80}")
            logging.info("Embedding and Final Layers:")
            logging.info(f"{'Layer':<25} {'Status':<8} {'Max Diff':<12} {'Mean Diff':<12} {'Rel Diff':<12} {'Shape'}")
            logging.info("-" * 80)
            
            for layer_name in ['embed_tokens', 'final_norm', 'lm_head']:
                if layer_name in comparison_results:
                    result = comparison_results[layer_name]
                    if result["status"] in ["MISSING", "SHAPE_MISMATCH"]:
                        logging.error(f"{layer_name:<25} {result['status']:<8} {result['error']}")
                    else:
                        status = result["status"]
                        max_diff = f"{result['max_absolute_diff']:.2e}"
                        mean_diff = f"{result['mean_absolute_diff']:.2e}"
                        rel_diff = f"{result['mean_relative_diff']:.2e}"
                        shape = str(result["shape"])
                        
                        log_func = logging.info if status == "PASS" else logging.warning
                        log_func(f"{layer_name:<25} {status:<8} {max_diff:<12} {mean_diff:<12} {rel_diff:<12} {shape}")
            
            # Then print transformer layers with fine-grained outputs
            logging.info(f"\n{'='*80}")
            logging.info("Transformer Layers (Fine-grained):")
            logging.info("-" * 80)
            
            # Determine number of layers by looking for layer_N_output keys
            num_layers = sum(1 for key in comparison_results.keys() if key.endswith("_output") and key.startswith("layer_"))
            
            for layer_idx in range(num_layers):
                logging.info(f"\nLayer {layer_idx}:")
                logging.info(f"{'Component':<25} {'Status':<8} {'Max Diff':<12} {'Mean Diff':<12} {'Rel Diff':<12} {'Shape'}")
                logging.info("-" * 60)
                
                # Order of components through the layer
                components = [
                    f"layer_{layer_idx}_input",
                    f"layer_{layer_idx}_ln1", 
                    f"layer_{layer_idx}_attn",
                    f"layer_{layer_idx}_ln2",
                    f"layer_{layer_idx}_ffn",
                    f"layer_{layer_idx}_output"
                ]
                
                for component in components:
                    if component in comparison_results:
                        result = comparison_results[component]
                        component_name = component.split(f"layer_{layer_idx}_")[1]
                        
                        if result["status"] in ["MISSING", "SHAPE_MISMATCH"]:
                            logging.error(f"  {component_name:<23} {result['status']:<8} {result['error']}")
                        else:
                            status = result["status"]
                            max_diff = f"{result['max_absolute_diff']:.2e}"
                            mean_diff = f"{result['mean_absolute_diff']:.2e}"
                            rel_diff = f"{result['mean_relative_diff']:.2e}"
                            shape = str(result["shape"])
                            
                            log_func = logging.info if status == "PASS" else logging.warning
                            log_func(f"  {component_name:<23} {status:<8} {max_diff:<12} {mean_diff:<12} {rel_diff:<12} {shape}")
            
            # Summary
            total_layers = len(comparison_results)
            passed_layers = sum(1 for r in comparison_results.values() if r["status"] == "PASS")
            
            logging.info(f"\nSummary for input {i+1}:")
            logging.info(f"Layers passed: {passed_layers}/{total_layers}")
            logging.info(f"Final output match: {final_output_close}")
            
            all_results[f"input_{i+1}"] = {
                "text": text,
                "comparison_results": comparison_results,
                "final_output_match": final_output_close,
                "max_output_diff": max_output_diff,
                "layers_passed": passed_layers,
                "total_layers": total_layers
            }
            
        except Exception as e:
            logging.error(f"Error during testing with input {i+1}: {str(e)}")
            all_results[f"input_{i+1}"] = {"error": str(e)}
        
        finally:
            # Clean up hooks
            hf_extractor.remove_hooks()
            custom_extractor.remove_hooks()
    
    # Overall summary
    logging.info(f"\n{'='*60}")
    logging.info("OVERALL TEST SUMMARY")
    logging.info(f"{'='*60}")
    
    total_tests = len([k for k in all_results.keys() if not "error" in all_results[k]])
    passed_tests = len([k for k, v in all_results.items() if not "error" in v and v["final_output_match"]])
    
    logging.info(f"Tests passed: {passed_tests}/{len(test_texts)}")
    
    for test_name, result in all_results.items():
        if "error" not in result:
            logging.info(f"{test_name}: {result['layers_passed']}/{result['total_layers']} layers passed, "
                        f"final output match: {result['final_output_match']}")
    
    return all_results


if __name__ == "__main__":
    hf_model, model = load_hf_model()
    
    # Run comprehensive testing
    test_results = test_model(hf_model, model)