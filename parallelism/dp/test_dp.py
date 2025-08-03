import copy
import torch
import torch.nn as nn
import torch.distributed as dist

from .toy_model import ToyModel
from .dp_naive import DDP, DDPOverlapBucketed, DDPOverlapIndividual, DDPOverlapIndividual2

from dataclasses import dataclass

@dataclass
class ModelConfig:
    in_features: int = 10
    out_features: int = 5


def get_batch(n_rows, model_config, device):
    # Set seed to ensure same x, y across all processes
    torch.manual_seed(42)
    x = torch.randn(n_rows, model_config.in_features).to(device)
    y = torch.randn(n_rows, model_config.out_features).to(device)
    return x, y

def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    global_rank = dist.get_rank()
    device = f"cuda:{global_rank}"
    epoch = 10
    batch_size = 32
    micro_bs = batch_size // dist.get_world_size()

    model_config = ModelConfig()
    x, y = get_batch(batch_size, model_config, device)

    model = ToyModel(in_features= model_config.in_features, out_features=model_config.out_features).to(device)
    ddp_model = DDPOverlapBucketed(copy.deepcopy(model), bucket_size_mb=1000)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    loss_fn = nn.MSELoss(reduction="mean")

    # DDP training loop
    for _ in range(epoch):
        ddp_optimizer.zero_grad()
        x_sharded = x[global_rank * micro_bs: (global_rank + 1) * micro_bs, :]
        y_sharded = y[global_rank * micro_bs: (global_rank + 1) * micro_bs, :]
        logits = ddp_model(x_sharded)
        loss = loss_fn(logits, y_sharded)
        loss.backward()
        ddp_model.finish_gradient_sync()
        ddp_optimizer.step()

    for _ in range(epoch):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    if global_rank == 0:
        for param, param_cp in zip(model.parameters(), ddp_model.parameters()):
            assert torch.allclose(param.data, param_cp.data)

if __name__ == "__main__":
    main()
    

