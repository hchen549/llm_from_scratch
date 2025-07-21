# Parallel Configuration Module

This module provides a `ParallelConfig` class for managing distributed training configurations with tensor parallelism (TP), pipeline parallelism (PP), and data parallelism (DP).

## Prerequisites

- PyTorch with distributed support
- NCCL backend for GPU communication
- Multiple GPUs (for multi-GPU setups)

## Running the Script

### Single Node, Multiple GPUs

To run the script with 4 GPUs on a single node:

```bash
torchrun --nproc_per_node=4 parallel_config.py
```

### Multiple Nodes

For multi-node training (example with 2 nodes, 4 GPUs each):

Node 1:
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 parallel_config.py
```

Node 2:
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<MASTER_IP> --master_port=29500 parallel_config.py
```

## Configuration

The script uses a configuration dictionary to specify parallelism dimensions:

```python
config = {"tp": 2, "pp": 1, "dp": 2}
```

- `tp`: Tensor Parallelism degree
- `pp`: Pipeline Parallelism degree  
- `dp`: Data Parallelism degree

**Important**: The product of `tp * pp * dp` must equal the total number of processes (world_size).

## Usage in Your Code

```python
from parallel_config import ParallelConfig

# Define your parallelism configuration
config = {
    "tp": 2,  # 2-way tensor parallelism
    "pp": 2,  # 2-way pipeline parallelism
    "dp": 1   # 1-way data parallelism
}

# Initialize parallel context
parallel_context = ParallelConfig(config)

# Access rank information
print(f"Global rank: {parallel_context.global_rank}")
print(f"TP rank: {parallel_context.tp_rank}")
print(f"PP rank: {parallel_context.pp_rank}")
print(f"DP rank: {parallel_context.dp_rank}")

# Set CUDA device
local_rank = parallel_context.get_local_rank()
torch.cuda.set_device(local_rank)
```

## Environment Variables

The script uses the following environment variables (automatically set by torchrun):
- `LOCAL_RANK`: Local rank for GPU assignment
- `RANK`: Global rank of the process
- `WORLD_SIZE`: Total number of processes

## Example Output

When running with 4 GPUs (tp=2, pp=1, dp=2):
```
Process 0: ParallelConfig(global_rank=0, world_size=4, tp=2[rank=0], pp=1[rank=0], dp=2[rank=0])
Process 0 using GPU 0
Process 1: ParallelConfig(global_rank=1, world_size=4, tp=2[rank=1], pp=1[rank=0], dp=2[rank=0])
Process 1 using GPU 1
Process 2: ParallelConfig(global_rank=2, world_size=4, tp=2[rank=0], pp=1[rank=0], dp=2[rank=1])
Process 2 using GPU 2
Process 3: ParallelConfig(global_rank=3, world_size=4, tp=2[rank=1], pp=1[rank=0], dp=2[rank=1])
Process 3 using GPU 3
```