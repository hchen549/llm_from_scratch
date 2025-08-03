import torch 
import torch.nn as nn
import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_sync(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

    
class DDPOverlapIndividual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.register_grad_hook()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_grad_hook(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._all_reduce)

    def _all_reduce(self, param):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def finish_gradient_sync(self):
        self.wait()
    
class DDPOverlapBucketed(nn.Module):
    
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_manager
