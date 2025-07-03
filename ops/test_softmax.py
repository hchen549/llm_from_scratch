from softmax import SoftmaxTriton

import torch

def get_random_input(input_shape):
    input = torch.rand(input_shape, device="cuda:0", requires_grad=True)
    dy = 0.1 * torch.randn_like(input, device="cuda:0", requires_grad=True)

    return input, dy

def test_softmax_fwd_triton():
    input_shape = (10, 20, 512)
    x, dy = get_random_input(input_shape)
    ground_truth = torch.softmax(x, dim=-1)
    triton_results = SoftmaxTriton.apply(x)

    torch.allclose(ground_truth, triton_results, atol = 1e-6)

# @pytest.mark.skipif(
#     not torch.cuda.is_available(),
#     reason="A GPU must be available to run Triton kernels",
# )
def test_softmax_backward_triton():
    input_shape = (10, 20, 512)
    x, dy = get_random_input(input_shape)
    torch.softmax(x, dim = -1).backward(dy)
    x_grad_ref = x.grad.clone()

    x.grad = None
    SoftmaxTriton.apply(x).backward(dy)

    assert torch.allclose(x.grad, x_grad_ref, rtol=1e-4, atol=1e-5), (
        x.grad,
        x_grad_ref,
    )

if __name__ == "__main__":
    # test_softmax_fwd_triton()
    test_softmax_backward_triton()


