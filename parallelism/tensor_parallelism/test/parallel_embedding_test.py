from ..mlp import TPEmbedding
from ..tensor_pp import setup_distributed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np

import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def tensor_parallel_embedding_test(num_embeddings, embedding_dim, parallel_config):
    rank = parallel_config['tp_rank']
    world_size = parallel_config['tp_size']
    device = f"cuda:{parallel_config['tp_rank']}"
    
    if parallel_config['tp_rank'] == 0:
        shared_weights = torch.randn(num_embeddings, embedding_dim, device=device)
    else:
        shared_weights = torch.empty(num_embeddings, embedding_dim, device=device)
    dist.broadcast(shared_weights, src=0)

    if rank == 0:
        x = torch.randint(0, num_embeddings, (32, 128), dtype=torch.long, device=device)
    else:
        x = torch.empty((32, 128), dtype=torch.long, device=device)
    dist.broadcast(x, src=0)
    
    # Initialize tensor parallel embedding
    tp_embedding = TPEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, parallel_config=parallel_config)
    tp_embedding.reset_parameters(weight=shared_weights)
    
    # Initialize raw embedding with the same shared weights
    raw_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    raw_embedding = raw_embedding.to(device)
    raw_embedding.weight.data.copy_(shared_weights)
    
    # Get regular embedding output
    y_tp = tp_embedding(x.clone())
    y_raw = raw_embedding(x.clone())

    # Compare the results
    torch.testing.assert_close(y_tp, y_raw, atol=1e-5, rtol=1e-5)
    logging.info(f"Test passed on rank {parallel_config['tp_rank']}!")

def main(rank, world_size):
    setup_distributed(rank=rank, world_size=world_size)
    parallel_config = {'tp_size': dist.get_world_size(), 'tp_rank': dist.get_rank()}
    tensor_parallel_embedding_test(10000, 10, parallel_config)
    
    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(main, args=(4,), nprocs=4)