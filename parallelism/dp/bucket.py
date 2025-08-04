import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List

class Bucket:

    def __init__(self, params: List[torch.nn.Parameter]) -> None:
        self.params = params
        self.bucket_size = 0

        for param in self.params:
            self.bucket_size += param.numel()
        
        self.grad_buffer = torch.zeros(size = (self.bucket_size,), device = self.params[0].device)
        self.ready_param_count = 0
                
    def sync_grad(self):
        self.handle = dist.all_reduce(self.grad_buffer, op = dist.ReduceOp.AVG, async_op=True)
    
    def wait(self):
        if self.handle is None:
            raise ValueError("No handle to wait for")
        return self.handle.wait()
    
    def reset(self):
        self.grad_buffer = torch.zeros(size = (self.bucket_size,), device = self.params[0].device)
        self.handle = None
        self.ready_param_count = 0
    



class BucketManager:
    def __init__(self, params: List[nn.Parameter], bucket_size_mb: float):
        self.params = params
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024  # Convert MB to bytes

        self.param_to_bucket_id = {}
        self.id_to_param = {}
        self.buckets = []
        self.bucket_to_ready = {}
       
        self._initialize_bucket()

    def _initialize_bucket(self):
        current_bucket_size = 0
        bucket_id = 0
        for param in self.params:
            if param.requires_grad == False:
                continue

            layer_size = param.numel() * param.element_size()  # Size in bytes
            if current_bucket_size + layer_size <= self.bucket_size_bytes:
                current_bucket_size += layer_size
            else:
                current_bucket_size = layer_size
                bucket_id += 1

            self.param_to_bucket_id[param] = bucket_id
            self.id_to_param[bucket_id] = self.id_to_param.get(bucket_id, []) + [param]

        for i in range(max(self.id_to_param.keys()) + 1): 
            bucket = Bucket(params=self.id_to_param[i])
            self.buckets.append(bucket)  

    def mark_param_as_ready(self, param):
        # Store the gradient for this parameter
        
        bucket_id = self.param_to_bucket_id[param]
        bucket = self.buckets[bucket_id]
        bucket.ready_param_count += 1

        if bucket.ready_param_count == len(bucket.params):
            offset = 0
            for param in bucket.params:
                num_elements = param.numel()
            
                bucket.grad_buffer[offset:offset + num_elements].copy_(param.grad.view(-1))
                offset += num_elements
            bucket.sync_grad()
           

    def wait(self):
        for bucket in self.buckets:
            bucket.wait()
            offset = 0
            for param in bucket.params:
                size = param.numel()
                param.grad.copy_(bucket.grad_buffer[offset: offset+size].view(param.shape))
                offset += size
            bucket.reset()
        



