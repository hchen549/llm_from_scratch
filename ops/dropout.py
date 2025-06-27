from os import umask
import torch
import torch.nn as nn

import triton
import triton.language as tl

@triton.jit
def dropout_fwd(x_ptr, x_mask_ptr, output_ptr, p, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x_ptrs = x_ptr + offsets
    x_mask_ptr = x_mask_ptr + offsets
    output_ptr = output_ptr + offsets
    mask = offsets < num_elements

    x = tl.load(x_ptrs, mask)
    x_mask = tl.load(x_mask, mask)
    output = tl.where(x_mask, x/1-p, 0)

    tl.store(output_ptr, output, mask)

class DropoutTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        assert x.is_contiguous()
        orig_shape = x.shape
      
        x_mask = (torch.rand_like(x) > p).to(torch.int32)
        output = torch.empty_like(x)

        num_elements = x.numel()
        
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]), )
        dropout_fwd[grid](x, x_mask, output, p, num_elements, BLOCK_SIZE = 1024 )
        output = output.view(orig_shape)
        return output
    
    def backward(ctx):
        pass