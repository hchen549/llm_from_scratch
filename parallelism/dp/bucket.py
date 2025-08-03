import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List

class Bucket:

    def __init__(self, params: List[torch.nn.Parameter]) -> None:
        self.params = params
        self.grad = None

    def concatenate(self):

        for param in self.params:
            if param.requires_grad:
                
    def sync_grad(self):
        dist.all_reduce(self.grad, op = dist.ReduceOp.SUM)

        

class BucketManager:

    def __init__(self, params: List[nn.Parameter], bucket_size_mb: float):

        self.params = params
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024  # Convert MB to bytes

        self.param_to_bucket_id = {}
        self.id_to_param = {}
        self.buckets = []

        self._initialize_bucket()

    def _initialize_bucket(self):
        current_bucket_size = 0
        bucket_id = 0
        for param in self.params:
            layer_size = param.numel() * param.element_size()  # Size in bytes
            if current_bucket_size + layer_size <= self.bucket_size_bytes:
                current_bucket_size += layer_size
            else:
                current_bucket_size = layer_size
                bucket_id += 1

            self.param_to_bucket_id[param] = bucket_id
            self.id_to_param[bucket_id] = self.id_to_param.get(bucket_id, []) + [param]

        for i in range(max(self.id_to_param.keys()) + 1):  # Fix off-by-one error
            bucket = Bucket(params=self.id_to_param[i])
            self.buckets.append(bucket)  # Fix syntax error
