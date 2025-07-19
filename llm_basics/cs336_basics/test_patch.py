import torch
import torch.nn as nn
from model_patch import patch_transformer_with_triton_rms_norm, _patch_rms_norm_module
from model import BasicsTransformerLM

# Simple RMSNorm module for testing
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # Original RMSNorm implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

def test_single_module_patch():
    """Test patching a single RMSNorm module."""
    print("Testing single module patch...")
    
    # Create a simple RMSNorm module
    hidden_size = 256
    rms_norm = RMSNorm(hidden_size=hidden_size, eps=1e-6)
    
    # Store original weight values
    original_weight = rms_norm.weight.data.clone()
    
    print(f"Before patching - Class: {rms_norm.__class__.__name__}")
    print(f"Before patching - eps: {rms_norm.eps}")
    
    # Patch the module
    _patch_rms_norm_module(rms_norm)
    
    print(f"After patching - Class: {rms_norm.__class__.__name__}")
    print(f"After patching - extra_repr: {rms_norm.extra_repr()}")
    
    # Verify weights are preserved
    assert torch.allclose(rms_norm.weight.data, original_weight), "Weights were not preserved!"
    
    # Test forward pass
    x = torch.randn(2, 10, hidden_size)
    output = rms_norm(x)
    print(f"Forward pass successful - output shape: {output.shape}")
    
    print("✓ Single module patch test passed!\n")

def test_transformer_patch():
    """Test patching an entire transformer model."""
    print("Testing transformer model patch...")
    
    # Create a small transformer model
    model = BasicsTransformerLM(
        vocab_size=1000,
        context_length=512,
        d_model=256,
        num_layers=2,
        num_heads=4,
        d_ff=1024,
        rope_theta=10000.0
    )
    
    # Count RMSNorm modules before patching
    rms_norm_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RMSNorm':
            rms_norm_count += 1
            print(f"Found RMSNorm: {name}")
    
    print(f"\nTotal RMSNorm modules found: {rms_norm_count}")
    
    # Patch the model
    model = patch_transformer_with_triton_rms_norm(model)
    
    # Verify all RMSNorm modules were patched
    patched_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RMSNorm_TritonPatched':
            patched_count += 1
            print(f"Patched module: {name} - {module.extra_repr()}")
    
    print(f"\nTotal modules patched: {patched_count}")
    assert patched_count == rms_norm_count, f"Expected {rms_norm_count} patched modules, but got {patched_count}"
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    try:
        output = model(input_ids)
        print(f"Forward pass successful - output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    
    print("✓ Transformer patch test passed!\n")

if __name__ == "__main__":
    print("Running patch tests...\n")
    test_single_module_patch()
    test_transformer_patch()
    print("All tests passed!")