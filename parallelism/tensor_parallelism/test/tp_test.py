import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensor_parallelism.mlp import ColumnParallelLinear, RowParallelLinear


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


def column_mlp_pass(x, y_label, model, optimizer):
    for _ in range(10):
        y_pred = model(x)
        loss = (y_pred - y_label).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.merge_weights()
    if dist.get_rank() == 0:
        print(f"model.weight.shape: {model.weight.shape}")

    return model

def init_models(in_features, out_features, parallel_config, device, tp_type):
    # non_tp_mlp = ColumnParallelLinear(in_features=in_features, out_features=out_features, parallel_config={'tp_size': 1, 'tp_rank': 0})
    if tp_type == 'column':
        print("Initializing column parallel MLP")
        tp_mlp = ColumnParallelLinear(in_features=in_features, out_features=out_features, parallel_config=parallel_config, device=device)
    elif tp_type == 'row':
        print("Initializing row parallel MLP")
        tp_mlp = RowParallelLinear(in_features=in_features, out_features=out_features, parallel_config=parallel_config, device=device)
    else:
        raise ValueError(f"Invalid tp_type: {tp_type}")

    # non_tp_mlp.reset_parameters()
    tp_mlp.reset_parameters()

    # return non_tp_mlp, tp_mlp
    return tp_mlp


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
    else:
        x = torch.empty(batch_size, in_features, device=device)
        y_label = torch.empty(batch_size, out_features, device=device)

    dist.broadcast(x, src=0)
    dist.broadcast(y_label, src=0)

    tp_mlp = init_models(in_features=in_features, out_features=out_features, parallel_config={'tp_size': world_size, 'tp_rank': rank}, device=device, tp_type='row')
    # optimizer = torch.optim.SGD(non_tp_mlp.parameters(), lr=0.01)
    optimizer_tp = torch.optim.SGD(tp_mlp.parameters(), lr=0.01)

    # column_mlp_pass(x, y_label, non_tp_mlp, optimizer)
    column_mlp_pass(x, y_label, tp_mlp, optimizer_tp)

    # if rank == 0:
    #     print("Comparing weights on rank 0...")
    #     print(f"Simple MLP weight norm: {non_tp_mlp.weight.norm().item()}")
    #     print(f"Column Parallel MLP (full) weight norm: {tp_mlp.weight.norm().item()}") # .weight is the full gathered weight
        
    #     all_close = torch.allclose(non_tp_mlp.weight, tp_mlp.weight, atol=1e-6)
    #     print(f"Weights are allclose: {all_close}")
    #     if not all_close:
    #         print("Difference:", (non_tp_mlp.weight - tp_mlp.weight).abs().max())
    #     assert all_close, "Model weights do not match after training!"
    #     print("Test passed on rank 0!")

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
