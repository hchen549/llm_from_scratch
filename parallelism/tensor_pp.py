import torch
import torch.nn as nn
import os
import torch.distributed as dist
import math

# setup_distributed, ColumnParallelLinear, simple_mlp_pass, column_parallel_mlp_pass
# ... (These functions remain the same as in the previous corrected code) ...
# Make sure setup_distributed is identical to the one that worked before for NCCL init
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank) # This sets the current CUDA device for this process
    return rank

# -------- add this helper --------
class _AllGatherGrad(torch.autograd.Function):
    """
    Forward  : all_gather along dim 1
    Backward : split the gradient and return the part that belongs to the
               local rank (no communication needed).
    """
    @staticmethod
    def forward(ctx, local_tensor, dim, group):
        ctx.dim   = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        ctx.world_size = world_size

        gather_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, local_tensor, group=group)
        return torch.cat(gather_list, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        # split and take the slice that corresponds to this rank
        rank = dist.get_rank(ctx.group)
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_chunks[rank].contiguous(), None, None

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, parallel_config: dict, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_config = parallel_config

        self.current_rank = parallel_config["rank"]
        self.world_size = parallel_config["world_size"]

        assert out_features % self.world_size == 0, "out_features must be divisible by world_size"
        self.out_features_per_shard = out_features // self.world_size

        # self.shard_weight is initialized on self.current_rank's CUDA device
        self.shard_weight = nn.Parameter(
            torch.empty(self.out_features_per_shard, self.in_features,
                        device=torch.device(f'cuda:{self.current_rank}'))
        )
        # self.weight buffer is also on self.current_rank's CUDA device
        self.register_buffer('weight', torch.empty(out_features, in_features,
                                                   device=torch.device(f'cuda:{self.current_rank}')))

        self.start_idx = self.current_rank * self.out_features_per_shard
        self.end_idx = self.start_idx + self.out_features_per_shard

        if init_weight is not None:
            with torch.no_grad():
                self.shard_weight.copy_(init_weight[self.start_idx:self.end_idx, :])
                self.weight.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.shard_weight, a=math.sqrt(5))
            # This gather is important if init_weight is None to populate the full self.weight
            # It requires all ranks to participate.
            if dist.is_initialized(): # Check if distributed is initialized before gathering
                 self.gather_weights()
            # If not distributed, self.weight would need different handling or remain uninit if starting from scratch for a single process

    def forward(self, x):
        shard_output = x @ self.shard_weight.t()
        full_output  = _AllGatherGrad.apply(              # differentiable
                           shard_output, 1, dist.group.WORLD)
        return full_output 

    def update_weights(self, learning_rate=0.01):
        with torch.no_grad():
            if self.shard_weight.grad is not None:
                self.shard_weight.data -= self.shard_weight.grad * learning_rate

    def gather_weights(self):
        shard_list = [torch.empty_like(self.shard_weight) for _ in range(self.world_size)]
        torch.distributed.all_gather(shard_list, self.shard_weight.contiguous()) # ensure contiguous
        full_weight_tensor = torch.cat(shard_list, dim=0)
        with torch.no_grad():
            self.weight.copy_(full_weight_tensor)


def simple_mlp_pass(x, y_label, in_features, out_features, parallel_config: dict, learning_rate=0.01, init_weight=None):
    device = torch.device(f'cuda:{parallel_config["rank"]}')
    model = nn.Linear(in_features, out_features, bias=False).to(device)
    if init_weight is not None:
        model.weight.data.copy_(init_weight) # init_weight is already on device

    for _ in range(10):
        model.zero_grad()
        y_pred = model(x)
        loss = (y_pred - y_label).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            model.weight.data -= model.weight.grad * learning_rate
    return model

def column_parallel_mlp_pass(x, y_label, in_features, out_features, parallel_config: dict, learning_rate=0.01, init_weight=None):
    # init_weight will be on the correct device when passed here
    model = ColumnParallelLinear(in_features, out_features, parallel_config, init_weight=init_weight)

    for _ in range(10):
        if model.shard_weight.grad is not None:
            model.shard_weight.grad.zero_()
        
        y_pred = model(x)
        loss = (y_pred - y_label).pow(2).mean()
        loss.backward()
        model.update_weights(learning_rate)
    model.gather_weights()
    return model

# ------------- CORRECTED compare_models FUNCTION -------------
def compare_models(rank, world_size):
    torch.manual_seed(42) # Seed for CPU operations
    torch.cuda.manual_seed(42) # Seed current CUDA device
    torch.cuda.manual_seed_all(42) # Seed all CUDA devices (good practice)

    local_rank = setup_distributed(rank, world_size) # Sets up NCCL and torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{local_rank}')

    batch_size = 4
    in_features = 20
    out_features = 10 
    assert out_features % world_size == 0, "out_features must be divisible by world_size"

    # Create tensors directly on the respective GPU device for each rank
    if rank == 0:
        x = torch.randn(batch_size, in_features, device=device)
        y_label = torch.randn(batch_size, out_features, device=device)
        init_weight = torch.randn(out_features, in_features, device=device)
    else:
        x = torch.empty(batch_size, in_features, device=device)
        y_label = torch.empty(batch_size, out_features, device=device)
        init_weight = torch.empty(out_features, in_features, device=device)

    dist.broadcast(x, src=0)
    dist.broadcast(y_label, src=0)
    dist.broadcast(init_weight, src=0)


    parallel_config = {"world_size": world_size, "rank": rank}

    # Pass clones of init_weight so modifications in one pass don't affect the other
    simple_mlp_model = simple_mlp_pass(x.clone(), y_label.clone(), in_features, out_features, parallel_config, init_weight=init_weight.clone())
    column_parallel_mlp_model = column_parallel_mlp_pass(x.clone(), y_label.clone(), in_features, out_features, parallel_config, init_weight=init_weight.clone())

    if rank == 0:
        print("Comparing weights on rank 0...")
        print(f"Simple MLP weight norm: {simple_mlp_model.weight.norm().item()}")
        print(f"Column Parallel MLP (full) weight norm: {column_parallel_mlp_model.weight.norm().item()}") # .weight is the full gathered weight
        
        all_close = torch.allclose(simple_mlp_model.weight, column_parallel_mlp_model.weight, atol=1e-6)
        print(f"Weights are allclose: {all_close}")
        if not all_close:
            print("Difference:", (simple_mlp_model.weight - column_parallel_mlp_model.weight).abs().max())
        assert all_close, "Model weights do not match after training!"
        print("Test passed on rank 0!")

    # Ensure all processes reach this point before destroying the group
    dist.barrier() # Good practice before destroy
    dist.destroy_process_group()
# ------------------------------------------------------------

def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA GPUs found. This example requires CUDA. Exiting.")
        return
   
    print(f"Running with world_size: {world_size}")
    torch.multiprocessing.spawn(compare_models, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()