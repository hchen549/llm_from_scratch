import torch
import triton
import triton.language as tl

class SoftmaxTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        n_rows, n_cols = x.shape
        output = torch.empty_like(x, device = x.device)
    
        ctx.block_size = triton.next_power_of_2(n_cols)

        softmax_triton_fwd[(n_rows,)](x, x.stride(0), output, n_cols, ctx.block_size)
        output = output.view(orig_shape)
        ctx.save_for_backward(output)
        return output

    def backward(ctx, grad_output):
        (output, ) = ctx.saved_tensors
        orig_shape = grad_output.shape
        grad_output = grad_output.view(-1, orig_shape[-1])
        n_rows, n_cols = grad_output.shape

        grad_x = torch.empty_like(grad_output)
        softmax_triton_bwd[(n_rows,)](grad_output, grad_output.stride(0), grad_x, output, n_cols, ctx.block_size)
        grad_x = grad_x.view(orig_shape)
        return grad_x

@triton.jit
def softmax_triton_fwd(x_ptr, x_stride, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_start = x_ptr + pid * x_stride
    output_start = output_ptr + pid * x_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x_ptrs = x_start + offsets
    output_ptrs = output_start + offsets
    
    x = tl.load(x_ptrs, mask, other= -float("inf"))

    max_x = tl.max(x)
    x = x - max_x
    exp_x = tl.exp(x)
    denominator = tl.sum(exp_x)

    output = exp_x/denominator
    tl.store(output_ptrs, output, mask)

@triton.jit
def softmax_triton_bwd(grad_output_ptr, grad_output_stride, grad_x_ptr, output_ptr, n_cols,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    grad_output_ptrs = grad_output_ptr + pid * grad_output_stride + offsets
    output_ptrs = output_ptr + pid * grad_output_stride + offsets
    grad_x_ptrs = grad_x_ptr + pid * grad_output_stride + offsets

    output= tl.load(output_ptrs, mask, other = -float('inf'))
    dy = tl.load(grad_output_ptrs, mask, other = 0)

    grad_x = dy * output -  output * tl.sum(dy * output)
    tl.store(grad_x_ptrs, grad_x, mask)


