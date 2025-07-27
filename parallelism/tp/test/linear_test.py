import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from parallelism.tp.mlp import ColumnParallelLinear, RowParallelLinear
import parallelism.parallel_config as pcfg

import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

@dataclass
class TestDataConfig:
    batch_size: int = 4
    in_features: int = 20
    out_features: int = 20

def column_mlp_pass(x, y_label, model, optimizer):
    for _ in range(10):
        y_pred = model(x)
        loss = (y_pred - y_label).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if hasattr(model, "merge_weights"):
        model.merge_weights()
    if dist.get_rank() == 0:
        logger.info(f"model.weight.shape: {model.weight.shape}")

    return model

def init_models(in_features, out_features, device, tp_type, shared_weight = None):
    # non_tp_mlp = ColumnParallelLinear(in_features=in_features, out_features=out_features, parallel_config={'tp_size': 1, 'tp_rank': 0})
    if tp_type == 'column':
        logger.info("Initializing column parallel MLP")
        tp_mlp = ColumnParallelLinear(in_features=in_features, out_features=out_features, device=device)
    elif tp_type == 'row':
        logger.info("Initializing row parallel MLP")
        tp_mlp = RowParallelLinear(in_features=in_features, out_features=out_features, device=device)
    else:
        raise ValueError(f"Invalid tp_type: {tp_type}")

    if shared_weight is not None:
        tp_mlp.load_weights(shared_weight)
    else:
        tp_mlp.reset_parameters()

    return tp_mlp


def compare_models(tp_type, config: TestDataConfig = None, seed = 42):
    if config is None:
        config = TestDataConfig()
        
    torch.manual_seed(seed) # Seed for CPU operations
    torch.cuda.manual_seed(seed) # Seed current CUDA device
    torch.cuda.manual_seed_all(seed) # Seed all CUDA devices (good practice)
    
    local_rank = pcfg.process_group_manager.global_rank
    world_size = pcfg.process_group_manager.world_size

    # local_rank = setup_distributed(rank, world_size) # Sets up NCCL and torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{local_rank}')

    assert config.out_features % world_size == 0, "out_features must be divisible by world_size"

    # Create tensors directly on the respective GPU device for each rank
    if local_rank == 0:
        x = torch.randn(config.batch_size, config.in_features, device=device)
        y_label = torch.randn(config.batch_size, config.out_features, device=device)
    else:
        x = torch.empty(config.batch_size, config.in_features, device=device)
        y_label = torch.empty(config.batch_size, config.out_features, device=device)

    dist.broadcast(x, src=0)
    dist.broadcast(y_label, src=0)

    non_tp_mlp = nn.Linear(in_features=config.in_features, out_features=config.out_features, bias = False).to(device=device)
    tp_mlp = init_models(in_features=config.in_features, out_features=config.out_features, device=device, tp_type=tp_type, shared_weight=non_tp_mlp.weight)
    optimizer = torch.optim.SGD(non_tp_mlp.parameters(), lr=0.01)
    optimizer_tp = torch.optim.SGD(tp_mlp.parameters(), lr=0.01)

    column_mlp_pass(x.clone(), y_label.clone(), non_tp_mlp, optimizer)
    column_mlp_pass(x.clone(), y_label.clone(), tp_mlp, optimizer_tp)

    if local_rank == 0:
        logger.info("Comparing weights on rank 0...")
        logger.info(f"Simple MLP weight norm: {non_tp_mlp.weight.norm().item()}")
        logger.info(f"Column Parallel MLP (full) weight norm: {tp_mlp.master_weight.norm().item()}") # .weight is the full gathered weight
        
        all_close = torch.allclose(non_tp_mlp.weight, tp_mlp.master_weight, atol=1e-6)
        logger.info(f"Weights are allclose: {all_close}")
        if not all_close:
            logger.error(f"Difference: {(non_tp_mlp.weight - tp_mlp.master_weight).abs().max()}")
        assert all_close, "Model weights do not match after training!"
        logger.info("Test passed on rank 0!")

    # Ensure all processes reach this point before destroying the group
    dist.barrier() # Good practice before destroy
    dist.destroy_process_group()
# ------------------------------------------------------------

def test_column_parallel():
    pcfg.setup_parallel_manager(config={"tp": 2})
    logger.info(f"world size {pcfg.process_group_manager.world_size}")
    compare_models(tp_type="column")

def test_row_parallel():
    pcfg.setup_parallel_manager(config={"tp": 4})
    logger.info(f"world size {pcfg.process_group_manager.world_size}")
    compare_models(tp_type="row", seed = 2025)

if __name__ == "__main__":
    test_column_parallel()
    # test_row_parallel()
