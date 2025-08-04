import torch 
import torch.nn as nn
import torch.distributed as dist

from .bucket import Bucket, BucketManager

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


class DDPOverlapIndividual2(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        self.params_with_grad = [param for param in self.module.parameters() if param.requires_grad]
        self.grad_buffer = [torch.zeros_like(param) for param in self.params_with_grad]

        self.register_grad_hook()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def sync_grad(self, index):
        def hook(grad):
            self.grad_buffer[index].data.copy_(grad)
            handle = dist.all_reduce(self.grad_buffer[index], op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
            
        return hook
    
    def register_grad_hook(self):
        for i, param in enumerate(self.params_with_grad):
            param.register_hook(self.sync_grad(index = i))
        
    def wait(self):
        for handle in self.handles:
            handle.wait()
        for param, sync_grad in zip(self.params_with_grad, self.grad_buffer):
            param.grad.data.copy_(sync_grad)

        self.grad_buffer = [torch.zeros_like(param) for param in self.params_with_grad]
        self.handles.clear()

    def finish_gradient_sync(self):
        self.wait()


    
class DDPOverlapBucketed(nn.Module):
    
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_manager = BucketManager(self.module.parameters(), bucket_size_mb)
        self.register_hook()

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)
    
    def register_hook(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(*unused):
            self.bucket_manager.mark_param_as_ready(param)
        return hook

    def finish_gradient_sync(self):
        self.bucket_manager.wait()
        
    

