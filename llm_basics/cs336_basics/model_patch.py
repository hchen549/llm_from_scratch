import torch
from torch import nn
from cs336_basics.model import BasicsTransformerLM, RMSNorm
from cs336_basics.model_triton import RMSNormTriton  # Assuming this is the correct path



def apply_triton_kernel(model, rms):
    if rms:
        replace_rms_norm_with_triton(model)
    return model

def replace_rms_norm_with_triton(model: nn.Module) -> BasicsTransformerLM:
    """
    Recursively finds all RMSNorm layers in a model and replaces them
    with a Triton-optimized version.

    Args:
        model: The model to patch.

    Returns:
        The patched model with Triton RMSNorm implementations.
    """
    for name, module in model.named_children():
        # print(name, model.type)
        # If the module is the one we want to replace
        if isinstance(module, RMSNorm):
            print(f"Replacing {name} with RMSNormTriton...")
            
            # 1. Create the new Triton module, preserving key attributes
            triton_module = RMSNormTriton(
                hidden_size=module.weight.shape[0],
                eps=module.eps
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            
            # 2. Copy the weights from the original module
            triton_module.weight.data.copy_(module.weight.data)
            
            # 3. Replace the original module with the new one
            setattr(model, name, triton_module)
            
        # If the module has children, recurse
        elif len(list(module.children())) > 0:
            replace_rms_norm_with_triton(module)
            
    return model

def main():
    """Example usage of the Triton RMS norm patching."""
    # Example model configuration
    model = BasicsTransformerLM(
        vocab_size=32000,
        context_length=2048,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0
    )
    
    print("Model before patching:")
    print(model)
    
    apply_triton_kernel(model, rms = True)

    print("\nModel after patching:")
    print(model) # You will see RMSNormTriton in the model definition now
    
    return model

if __name__ == "__main__":
    main()