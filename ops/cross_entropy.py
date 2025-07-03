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
def chunked_cross_entropy_fwd(logits_ptr, logits_stride, loss_ptr, label_ptr, logsumexp_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    logits_ptr += logits_stride * pid
    loss_ptr += pid
    label_ptr += pid
    logsumexp_ptr += pid

    label = tl.load(label_ptr)

    offsets = tl.arange(0, BLOCK_SIZE)
    row_max = -float("inf")
    sumexp = 0.0
    # similar to online softmax, use the first iteration to compute the sumexp
    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + offsets < n_cols
        logits_block = tl.load(logits_ptr + block_start + offsets, mask, other = -float("inf"))
        block_max = tl.max(logits_block, axis = -1)
        prev_max = row_max
        row_max = tl.maximum(prev_max, block_max)
        sumexp = tl.sum(tl.exp(logits_block - row_max)) + sumexp * tl.exp(prev_max - row_max)

    loss = 0.0
    if label != -100:
        xi = tl.load(logits_ptr + label)
        loss = row_max + tl.log(sumexp) - xi
    
    tl.store(logsumexp_ptr, tl.log(sumexp) + row_max)
    tl.store(loss_ptr, loss)

@triton.jit
def inplace_chunked_cross_entropy_fwd(logits_ptr, logits_stride, loss_ptr, label_ptr, logsumexp_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    logits_ptr += logits_stride * pid
    loss_ptr += pid
    label_ptr += pid
    logsumexp_ptr += pid

    label = tl.load(label_ptr)

    offsets = tl.arange(0, BLOCK_SIZE)
    row_max = -float("inf")
    sumexp = 0.0
    # similar to online softmax, use the first iteration to compute the sumexp
    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + offsets < n_cols
        logits_block = tl.load(logits_ptr + block_start + offsets, mask, other = -float("inf"))
        block_max = tl.max(logits_block, axis = -1)
        prev_max = row_max
        row_max = tl.maximum(prev_max, block_max)
        sumexp = tl.sum(tl.exp(logits_block - row_max)) + sumexp * tl.exp(prev_max - row_max)

    loss = 0.0
    logsumexp = tl.log(sumexp) + row_max
    if label != -100:
        xi = tl.load(logits_ptr + label)
        loss = logsumexp - xi
    tl.store(loss_ptr, loss)

    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + offsets < n_cols
        x = tl.load(logits_ptr + block_start + offsets, mask, other = 0.0)
        dx = tl.exp(x - logsumexp)
        dx = tl.where(label == block_start + offsets, dx - 1, dx)
        tl.store(logits_ptr + block_start + offsets, dx, mask)
    
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

@triton.jit
def chunked_cross_entropy_bwd(dy_ptr, label_ptr, logsumexp_ptr, x_ptr,  dx_ptr, dx_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_ptr += dx_stride * pid
    dx_ptr += dx_stride * pid
    dy_ptr += pid
    label_ptr += pid
    logsumexp_ptr += pid

    label = tl.load(label_ptr)
    logsumexp = tl.load(logsumexp_ptr)
    offsets = tl.arange(0, BLOCK_SIZE)

    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + offsets < n_cols
        x = tl.load(x_ptr + block_start + offsets, mask, other = 0.0)
        dx = tl.exp(x - logsumexp)
        dx = tl.where(label == block_start + offsets, dx - 1, dx)
        tl.store(dx_ptr + block_start + offsets, dx, mask)



class CrossEntropyTriton(torch.autograd.Function):
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor, reduction = "mean", inplace = False):
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.flatten()

        n_rows, n_cols = logits.shape
        loss = torch.empty(n_rows, dtype=logits.dtype, device=logits.device)
        logsumexp = torch.empty(n_rows, dtype=logits.dtype, device=logits.device)

        ctx.block_size = 256
        ctx.inplace = inplace
        if ctx.block_size >= n_cols:
            print("using non-chunked cross entropy")
            cross_entropy_fwd[(n_rows, )](logits, logits.stride(0), loss, labels, logsumexp, n_cols, ctx.block_size)
        else:
            if ctx.inplace == True:
                print("using inplace chunked cross entropy")
                inplace_chunked_cross_entropy_fwd[(n_rows, )](logits, logits.stride(0), loss, labels, logsumexp, n_cols, ctx.block_size)
            else:
                print("using chunked cross entropy")
                chunked_cross_entropy_fwd[(n_rows, )](logits, logits.stride(0), loss, labels, logsumexp, n_cols, ctx.block_size)

        ctx.save_for_backward(logits, labels, logsumexp)
        ctx.reduction = reduction
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    def backward(ctx, dloss):
        logits, labels, logsumexp = ctx.saved_tensors
        if ctx.inplace == True:
            print("using in place chunked cross entropy bwd")
            return logits, None, None, None
        
        grad_logits = torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        if ctx.reduction == "mean":
            dloss *= n_cols # un-reduce the loss

        if ctx.block_size >= n_cols:
            print("using non-chunked cross entropy bwd")
            cross_entropy_bwd[(n_rows,)](dloss, labels, logsumexp, logits,  grad_logits, grad_logits.stride(0), n_cols, ctx.block_size)
        else:
            print("using chunked cross entropy bwd")
            chunked_cross_entropy_bwd[(n_rows,)](dloss, labels, logsumexp, logits,  grad_logits, grad_logits.stride(0), n_cols, ctx.block_size)

        return grad_logits, None, None, None