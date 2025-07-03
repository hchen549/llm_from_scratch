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

        ctx.block_size = 256
        if ctx.block_size < n_cols:
        # ctx.block_size = triton.next_power_of_2(n_cols)
            print("use online softmax")
            online_softmax_triton_fwd[(n_rows, )](x, x.stride(0), output, n_cols, ctx.block_size)
        else:
            print("use normal softmax")
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
        if ctx.block_size < n_cols:
            print("use online softmax bwd")
            online_softmax_triton_bwd[(n_rows,)](grad_output, grad_output.stride(0), grad_x, output, n_cols, ctx.block_size)
        else:
            print("use normal softmax bwd")
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
def online_softmax_triton_fwd(x_ptr, x_stride, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * x_stride
    output_ptr += pid * x_stride
    curr_max = -float("inf")
    denominator_sum = 0.0

    # first iteration: compute the denominator_sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = block_start + offsets < n_cols

        x_block = tl.load(row_ptr + block_start + offsets, mask, -float("inf"))
        block_max = tl.max(x_block, axis = -1)
        prev_max = curr_max
        curr_max = tl.maximum(prev_max, block_max)
        block_sum = tl.sum(tl.exp(x_block - curr_max))
        denominator_sum = denominator_sum * tl.exp(prev_max - curr_max) + block_sum

    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = block_start + offsets < n_cols

        x_block = tl.load(row_ptr + block_start + offsets, mask, -float("inf"))
        output = tl.exp(x_block - curr_max)/denominator_sum
        tl.store(output_ptr + block_start + offsets, output, mask)

    


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

@triton.jit
def online_softmax_triton_bwd(dy_ptr, dy_stride, dx_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    dy_ptr += pid * dy_stride
    dx_ptr += pid * dy_stride
    y_ptr += pid * dy_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    block_sum = 0.0

    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + offsets < n_cols
        dy_block = tl.load(dy_ptr + block_start + offsets, mask, 0)
        y_block = tl.load(y_ptr + block_start + offsets, mask, 0)
        block_sum += tl.sum(dy_block * y_block, axis = -1)

    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + offsets < n_cols
        dy_block = tl.load(dy_ptr + block_start + offsets, mask, 0)
        y_block = tl.load(y_ptr + block_start + offsets, mask, 0)

        dx_block = dy_block * y_block - y_block * block_sum
        tl.store(dx_ptr + block_start + offsets, dx_block, mask )

