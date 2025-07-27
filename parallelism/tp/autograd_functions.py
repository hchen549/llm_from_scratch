import torch
import torch.nn as nn
import torch.distributed as dist

import parallelism.parallel_config as pcfg

class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        if pcfg.process_group_manager.tp > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group = pcfg.process_group_manager.tp_group)
        return grad_output
    

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if pcfg.process_group_manager.tp > 1:
            tensor_list = [torch.empty_like(x) for _ in range(pcfg.process_group_manager.tp)]
            dist.all_gather(tensor_list, x, group = pcfg.process_group_manager.tp_group)
            return torch.cat(tensor_list, dim=-1) # (bs, out_features/n) -> (bs, out_features)
        else:
            return x
    
    @staticmethod
    def backward(ctx, grad_output):
        if pcfg.process_group_manager.tp > 1:
            assert grad_output.shape[1] % pcfg.process_group_manager.tp == 0, "grad_output must be divisible by tp_size"
            grad_chunk_list = torch.split(grad_output, grad_output.shape[1] // pcfg.process_group_manager.tp, dim = -1)
            return grad_chunk_list[pcfg.process_group_manager.tp_rank].contiguous()
        else:
            return grad_output
        
class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x: (..., in_features)
        if dist.get_world_size() > 1:
            assert x.shape[-1] % pcfg.process_group_manager.tp == 0, "x must be divisible by tp_size"
            x_chunk_list = torch.split(x, x.shape[-1] // pcfg.process_group_manager.tp , dim = -1)
            x_chunk = x_chunk_list[pcfg.process_group_manager.tp_rank].contiguous()
            return x_chunk
        
        else:
            return x

    @staticmethod
    def backward(ctx, grad_output):
        if dist.get_world_size() > 1:
            grad_output_chunk_list = [torch.empty_like(grad_output) for _ in range(pcfg.process_group_manager.tp)]
            dist.all_gather(grad_output_chunk_list, grad_output, group = pcfg.process_group_manager.tp_group)
            return torch.cat(grad_output_chunk_list, dim=-1) # (bs, in_features/n) -> (bs, in_features)
        else:
            return grad_output
    
class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x: (bs, out_features)
        if pcfg.process_group_manager.tp > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group = pcfg.process_group_manager.tp_group)
            return x
        else:
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    