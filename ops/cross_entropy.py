import triton
import triton.language as tl

import torch.nn


@triton.jit
def cross_entropy_fwd(logits_ptr, logits_stride, loss_ptr, label_ptr, logsumexp_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    loss_ptr += pid
    label_ptr += pid
    logsumexp_ptr += pid

    offsets = tl.arange(0, BLOCK_SIZE)
    logits_ptrs = logits_ptr + pid * logits_stride + offsets 
    mask = offsets < n_cols

    logits = tl.load(logits_ptrs, mask, other = -float("inf"))
    label_idx = tl.load(label_ptr).to(tl.int32)
    row_max = tl.max(logits, axis = -1)
    logsumexp = row_max + tl.log(tl.sum(tl.exp(logits - row_max), axis = -1))

    if label_idx != -100:
        curr_x = tl.load(logits_ptr + pid * logits_stride + label_idx)
        loss = logsumexp - curr_x
    else:
        loss = 0.0

    tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)

@triton.jit
def cross_entropy_bwd(dloss_ptr, labels_ptr, logsumexp_ptr, logits_ptr,  grad_logits_ptr, grad_input_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    dloss_ptr = dloss_ptr + pid
    labels_ptr = labels_ptr + pid
    logsumexp_ptr = logsumexp_ptr + pid

    dloss = tl.load(dloss_ptr)
    logsumexp = tl.load(logsumexp_ptr)
    label_idx = tl.load(labels_ptr)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    logits_ptrs = logits_ptr + pid * grad_input_stride + offsets
    grad_logits_ptrs = grad_logits_ptr + pid * grad_input_stride + offsets
    logits = tl.load(logits_ptrs, mask = mask, other = -float("inf"))
    
    dlogits = tl.exp(logits - logsumexp)
    dlogits = tl.where(offsets == label_idx, dlogits - 1, dlogits)

    tl.store(grad_logits_ptrs, dloss * dlogits, mask = mask)

class CrossEntropyTriton(torch.autograd.Function):
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.flatten()

        n_rows, n_cols = logits.shape
        loss = torch.empty(n_rows, dtype=logits.dtype, device=logits.device)
        logsumexp = torch.empty(n_rows, dtype=logits.dtype, device=logits.device)

        ctx.block_size = triton.next_power_of_2(n_cols)
        cross_entropy_fwd[(n_rows, )](logits, logits.stride(0), loss, labels, logsumexp, n_cols, ctx.block_size)

        ctx.save_for_backward(logits, labels, logsumexp)

        return loss.mean()

    def backward(ctx, dloss):
        
        logits, labels, logsumexp = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        dloss *= n_cols # un-reduce the loss

        cross_entropy_bwd[(n_rows,)](dloss, labels, logsumexp, logits,  grad_logits, grad_logits.stride(0), n_cols, ctx.block_size)

        return grad_logits, None