import torch
import triton
import triton.language as tl

class RmsNormPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps = 1e-5):
        # (B, L, H), (B, L)
        return x/torch.sqrt(torch.sum(x**2, dim = -1, keepdim = True)/x.shape[-1] + eps) * weight

    @staticmethod
    def backward(ctx, grad_output):
        pass
    

class RmsNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps = 1e-5):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).contiguous()
        n_rows, n_cols = x.shape
        output = torch.empty_like(x, device=x.device)
        ctx.block_size = triton.next_power_of_2(n_cols)
        rms_norm_fwd[(n_rows, )](x, weight, x.stride(0), eps, output, output.stride(0), n_cols, ctx.block_size)
        ctx.save_for_backward(x, weight)
        output = output.view(x_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output, eps = 1e-5):
        orig_shape = grad_output.shape
        x, weight = ctx.saved_tensors
        grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()
        n_rows, n_cols = grad_output.shape

        grad_x = torch.empty_like(x, device = x.device)
        partial_grad_weight = torch.empty_like(x, device=x.device)
        
        rms_norm_bwd[(n_rows, )](grad_output, grad_x, partial_grad_weight, x, weight, x.stride(0), eps, n_cols, ctx.block_size)

        grad_weight = partial_grad_weight.sum(dim = 0)
        grad_x = grad_x.view(orig_shape)

        return grad_x, grad_weight, None

@triton.jit
def rms_norm_fwd(x_ptr, weight_ptr, x_stride, eps, output_ptr, output_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)

    row_start_ptr = x_ptr + row_idx * x_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)

    x_squared = tl.sum(x * x) / n_cols
    x_norm = tl.sqrt(x_squared + eps)
    x_norm_inv = 1 / x_norm
    x_normed = x * x_norm_inv
    output = x_normed * weight
    
    output_row_ptr = output_ptr + row_idx * output_stride
    tl.store(output_row_ptr + offsets, output, mask=mask)

@triton.jit
def rms_norm_bwd(grad_output_ptr, grad_x_ptr, partial_grad_weight_ptr, x_ptr, weight_ptr, x_stride, eps, n_cols, BLOCK_SIZE: tl.constexpr ):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # load input
    grad_output_ptrs = grad_output_ptr + x_stride * row_idx +  offsets # (B, H)
    x_ptrs = x_ptr + x_stride * row_idx +  offsets # (B, H)
    weight_ptrs = weight_ptr + offsets # (H,)
   
    grad_output = tl.load(grad_output_ptrs, mask, other = 0.0)
    x = tl.load(x_ptrs, mask, other=0.0)
    weight = tl.load(weight_ptrs, mask, other = 0.0)

    # output
    partial_grad_weight_ptrs = partial_grad_weight_ptr + x_stride * row_idx + offsets # (B, H)
    grad_x_ptrs = grad_x_ptr + x_stride * row_idx + offsets # (B, H)
    
     # recompute rms
    rms = tl.sqrt(tl.sum(x * x)/n_cols + eps)

    # gradient with respect to weight
    grad_weight_row = grad_output * x * 1/rms 
    tl.store(partial_grad_weight_ptrs, grad_weight_row, mask=mask)

    # gradient with respect to x
    corrector = tl.sum(weight * x * grad_output)
    grad_x_row = grad_output * weight * 1/rms - corrector * x/(n_cols * rms * rms * rms)
    tl.store(grad_x_ptrs, grad_x_row, mask = mask)


    



    




