import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tensor_parallelism.autograd_functions import Copy, Gather, Scatter, Reduce

class ColumnParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, parallel_config: dict, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = parallel_config['tp_size']
        self.tp_rank = parallel_config['tp_rank']
        self.device = device

        assert out_features % self.tp_size == 0, "out_features must be divisible by tp_size"
        out_features_per_rank = out_features // self.tp_size

        self.weight = nn.Parameter(torch.randn(out_features_per_rank, in_features, device=self.device))

    def forward(self, x):
        x = Copy.apply(x)
        x = F.linear(x, self.weight)
        x = Gather.apply(x)
        return x
    
    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight_partition, a=math.sqrt(5))

    def reset_parameters(self):
        weight = nn.Parameter(torch.randn(self.out_features, self.in_features, device=self.device))
        dist.broadcast(weight, src=0)
        # weight_partioned = torch.split(weight, self.out_features // self.tp_size, dim=-1)
        
        columns_per_rank = self.out_features // self.tp_size
        self.weight.data.copy_(weight[columns_per_rank * self.tp_rank:columns_per_rank * (self.tp_rank + 1), :])

    def merge_weights(self):
        weights_list = [torch.empty_like(self.weight) for _ in range(dist.get_world_size())]
        dist.all_gather(weights_list, self.weight)
        self.weight = nn.Parameter(torch.cat(weights_list, dim=0))   # (out_features/n, in_features) -> (out_features, in_features)
        return self.weight
    
class RowParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, parallel_config: dict, device):
        super().__init__()
        
        self.tp_size = parallel_config['tp_size']
        self.tp_rank = parallel_config['tp_rank']
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        assert in_features % self.tp_size == 0, "out_features must be divisible by tp_size"
        self.in_features_per_rank = in_features // self.tp_size

        self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_rank, device=device))
        

    def forward(self, x):
        x = Scatter.apply(x)
        x = F.linear(x, self.weight)
        x = Reduce.apply(x)
        return x
    
    def reset_parameters(self):
        weight = nn.Parameter(torch.randn(self.out_features, self.in_features, device=self.device))
        dist.broadcast(weight, src=0)   
        self.weight.data.copy_(weight[:, self.in_features_per_rank * self.tp_rank:self.in_features_per_rank * (self.tp_rank + 1)])

    def merge_weights(self):
        weights_list = [torch.empty_like(self.weight) for _ in range(self.tp_size)]
        dist.all_gather(weights_list, self.weight)

        self.weight = nn.Parameter(torch.cat(weights_list, dim=-1)) # (out_features, in_features/n) -> (out_features, in_features)
        return self.weight