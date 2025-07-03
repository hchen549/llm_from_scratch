import torch
import torch.nn as nn
from cross_entropy import CrossEntropyTriton

def get_test_data(shape, device = None, dtype = None):
    input = torch.randn(shape, device = device, dtype=dtype)
    labels = torch.randint(low=0, high = shape[-1], size = (shape[0],), device = device, dtype=dtype)
    print("input shape", input.shape)
    print("labels shape", labels.shape)

    return input, labels

def test_cross_entropy_fwd(shape, device=None, dtype=None):
    input, labels = get_test_data(shape, device, dtype)
    cross_entropy = CrossEntropyTriton()
    expected_output = nn.CrossEntropyLoss()(input, labels)
    custom_output = cross_entropy.apply(input, labels, "mean", True)

    print("custom_output", custom_output)
    print("custom_output mean", custom_output.mean())
    print("expected output", expected_output)

    torch.allclose(custom_output.mean(), expected_output, atol = 1e-10)

def test_cross_entropy_bwd(shape, device=None, dtype=None):
    input, labels = get_test_data(shape, device, dtype)
    input_expected = input.clone().requires_grad_(True)

    cross_entropy = CrossEntropyTriton()
    nn.CrossEntropyLoss()(input_expected, labels).backward()

    input_custom = input.clone().requires_grad_(True)
    cross_entropy.apply(input_custom, labels, "mean", True).backward()

    torch.allclose(input_expected.grad, input_custom.grad, atol = 1e-5)

def main():
    shape = (16 * 24, 512)
    test_cross_entropy_fwd(shape, device = "cuda:0")
    test_cross_entropy_bwd(shape, device = "cuda:0")

if __name__ == "__main__":
    main()


