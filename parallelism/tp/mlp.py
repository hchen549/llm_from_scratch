import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from parallelism.tp.autograd_functions import Copy, Gather, Scatter, Reduce

import parallelism.parallel_config as pcfg

class ColumnParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, device, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = pcfg.process_group_manager.tp
        self.tp_rank = pcfg.process_group_manager.tp_rank
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
        self.master_weight = nn.Parameter(torch.cat(weights_list, dim=0))   # (out_features/n, in_features) -> (out_features, in_features)
        return self.master_weight
    
    def load_weights(self, shared_weight):
        """Load weights from a shared (non-partitioned) weight tensor.
        
        Args:
            shared_weights: Full weight tensor of shape (out_features, in_features)
        """
        # Handle both tensor and Parameter inputs
        weight_data = shared_weight.data if hasattr(shared_weight, 'data') else shared_weight
        
        expected_shape = (self.out_features, self.in_features)
        if weight_data.shape != expected_shape:
            raise ValueError(f"Expected weight shape {expected_shape}, got {weight_data.shape}")

        partition_size = self.out_features // self.tp_size
        
        start_idx = partition_size * self.tp_rank
        end_idx = partition_size * (self.tp_rank + 1)

        self.weight.data.copy_(weight_data[start_idx:end_idx, :])
    
class RowParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, device, scatter_input=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = pcfg.process_group_manager.tp
        self.tp_rank = pcfg.process_group_manager.tp_rank
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

        self.master_weight = nn.Parameter(torch.cat(weights_list, dim=-1)) # (out_features, in_features/n) -> (out_features, in_features)
        return self.master_weight

    def load_weights(self, shared_weight):
        weight_data = shared_weight.data if hasattr(shared_weight, "data") else shared_weight

        assert weight_data.shape == (self.out_features, self.in_features), "miss match in weight shape"

        partition_size = self.in_features // self.tp_size
        start_idx = partition_size * self.tp_rank
        end_idx = partition_size * self.tp_rank + partition_size

        self.weight.data.copy_(weight_data[:, start_idx:end_idx])
    
class TPEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = pcfg.process_group_manager.tp
        self.tp_rank = pcfg.process_group_manager.tp_rank
        self.start_index = self.num_embeddings // self.tp_size * self.tp_rank
        self.end_index = self.num_embeddings // self.tp_size * (self.tp_rank + 1)
        self.device = "cuda:" + str(pcfg.process_group_manager.global_rank)
        self.weight = nn.Parameter(torch.randn(self.num_embeddings//self.tp_size, self.embedding_dim, device=self.device))

    def forward(self, x): # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        # Create a mask for indices that belong to this rank's partition
        mask = (x >= self.start_index) & (x < self.end_index)
        x[mask], x[~mask] = x[mask] - self.start_index, 0.0
        
        output = F.embedding(x, self.weight) # (batch_size, seq_len, embedding_dim)
        # Zero out embeddings for indices not belonging to this rank
        mask_expanded = mask.unsqueeze(-1).expand_as(output)
        output[~mask_expanded] = 0.0
        output = Reduce.apply(output)
        return output
        
    def reset_parameters(self, weight=None):
        if weight is None:
            weight = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim, device=self.device))
            dist.broadcast(weight, src=0)
            self.weight.data.copy_(weight[self.start_index:self.end_index, :]) # (vocab_size//n, embedding_dim) 
        else:
            assert weight.shape == (self.num_embeddings, self.embedding_dim), "Weight shape must match"
            self.weight.data.copy_(weight[self.start_index:self.end_index, :])

    def merge_weights(self):
        weights_list = [torch.empty_like(self.weight) for _ in range(self.tp_size)]
        dist.all_gather(weights_list, self.weight)
        self.weight = nn.Parameter(torch.cat(weights_list, dim=0)) # (vocab_size//n, embedding_dim) -> (vocab_size, embedding_dim)
        return self.weight