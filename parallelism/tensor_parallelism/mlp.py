import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tensor_parallelism.autograd_functions import Copy, Gather, Scatter, Reduce

class ColumnParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, parallel_config: dict, device, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = parallel_config['tp_size']
        self.tp_rank = parallel_config['tp_rank']
        self.device = device
        self.gather_output = gather_output

        assert out_features % self.tp_size == 0, "out_features must be divisible by tp_size"
        out_features_per_rank = out_features // self.tp_size

        self.weight = nn.Parameter(torch.randn(out_features_per_rank, in_features, device=self.device))

    def forward(self, x):
        x = Copy.apply(x)
        x = F.linear(x, self.weight)
        if self.gather_output:
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

    def __init__(self, in_features, out_features, parallel_config: dict, device, scatter_input=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = parallel_config['tp_size']
        self.tp_rank = parallel_config['tp_rank']
        self.device = device
        self.scatter_input = scatter_input
        
        assert in_features % self.tp_size == 0, "out_features must be divisible by tp_size"
        self.in_features_per_rank = in_features // self.tp_size

        self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_rank, device=device))
        

    def forward(self, x):
        if self.scatter_input:
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
    
class TPEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, parallel_config: dict, device):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = parallel_config['tp_size']
        self.tp_rank = parallel_config['tp_rank']
        self.start_index = self.num_embeddings // self.tp_size * self.tp_rank
        self.end_index = self.num_embeddings // self.tp_size * (self.tp_rank + 1)
        self.device = device

    def forward(self, x): # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        mask = (x >= self.start_index) & (x < self.end_index)
        x[mask], x[~mask] = x[mask] - self.start_index, 0
        x = F.embedding(x, self.weight) # (batch_size, seq_len, embedding_dim)
        mask_expanded = mask.unsqueeze(-1)
        x[~mask_expanded] = 0.0
        x = Reduce.apply(x)
        return x    
        
    def reset_parameters(self):
        weight = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim, device=self.device))
        dist.broadcast(weight, src=0)
        self.weight.data.copy_(weight[self.start_index:self.end_index, :]) # (vocab_size//n, embedding_dim) 

    def merge_weights(self):
        weights_list = [torch.empty_like(self.weight) for _ in range(self.tp_size)]
        dist.all_gather(weights_list, self.weight)
        self.weight = nn.Parameter(torch.cat(weights_list, dim=0)) # (vocab_size//n, embedding_dim) -> (vocab_size, embedding_dim)
        return self.weight