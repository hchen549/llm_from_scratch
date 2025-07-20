import torch
import torch.distributed as dist
from typing import Dict, Optional
import os

class ParallelConfig:
    def __init__(self, config: Dict):
        self.tp = config.get("tp", 1)
        self.pp = config.get("pp", 1)
        self.dp = config.get("dp", 1)
        
        # Initialize distributed if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        assert self.world_size == self.tp * self.pp * self.dp, f"World size ({self.world_size}) != DP ({self.dp_size}) * PP ({self.pp_size}) * TP ({self.tp_size})"

        self.grid = torch.arange(0, self.world_size).view(self.dp, self.pp, self.tp)
        
        # Validate configuration
        assert self.tp * self.pp * self.dp == self.world_size, \
            f"Product of tp({self.tp}) * pp({self.pp}) * dp({self.dp}) must equal world_size({self.world_size})"

        # Calculate rank within each parallelism dimension
        self.tp_rank = self.global_rank % self.tp
        self.pp_rank = (self.global_rank // self.tp) % self.pp
        self.dp_rank = self.global_rank // (self.tp * self.pp)

        self.tp_group = self.grid[self.dp_rank, self.pp_rank, :].tolist()
        self.pp_group = self.grid[self.dp_rank, :, self.tp_rank].tolist()
        self.dp_group = self.grid[:, self.pp_rank, self.tp_rank].tolist()
        
    

    def __repr__(self):
        return (f"ParallelConfig(global_rank={self.global_rank}, world_size={self.world_size}, "
                f"tp={self.tp}[rank={self.tp_rank}], pp={self.pp}[rank={self.pp_rank}], "
                f"dp={self.dp}[rank={self.dp_rank}])")
    
    def get_local_rank(self) -> int:
        """Get the local rank for CUDA device assignment"""
        return int(os.environ.get('LOCAL_RANK', 0))
    

if __name__ == "__main__":
   
    config = {"tp": 2, "pp": 1, "dp": 2}
    
    try:
        parallel_context = ParallelConfig(config)
        print(f"Process {parallel_context.global_rank}: {parallel_context}")
        
        # Set CUDA device based on local rank
        local_rank = parallel_context.get_local_rank()
        torch.cuda.set_device(local_rank)
        print(f"Process {parallel_context.global_rank} using GPU {local_rank}")
        
    except Exception as e:
        print(f"Error initializing parallel config: {e}")

