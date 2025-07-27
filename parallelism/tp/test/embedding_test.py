from ..mlp import TPEmbedding
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np

import logging

import parallelism.parallel_config as pcfg


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def tensor_parallel_embedding_test(num_embeddings, embedding_dim,):
    rank = pcfg.process_group_manager.tp_rank
    device = f"cuda:{rank}"
    logger.info(f"[Rank {rank}] Starting tensor_parallel_embedding_test")
    logger.info(f"[Rank {rank}] Device: {device}")
    
    if pcfg.process_group_manager.tp_rank  == 0:
        shared_weights = torch.randn(num_embeddings, embedding_dim, device=device)
    else:
        shared_weights = torch.empty(num_embeddings, embedding_dim, device=device)
    dist.broadcast(shared_weights, src=0)

    if rank == 0:
        x = torch.randint(0, num_embeddings, (32, 128), dtype=torch.long, device=device)
    else:
        x = torch.empty((32, 128), dtype=torch.long, device=device)
    dist.broadcast(x, src=0, group=pcfg.process_group_manager.tp_group)
    
    # Initialize tensor parallel embedding
    tp_embedding = TPEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    tp_embedding.reset_parameters(weight=shared_weights)
    
    # Initialize raw embedding with the same shared weights
    raw_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    raw_embedding = raw_embedding.to(device)
    raw_embedding.weight.data.copy_(shared_weights)
    
    # Get regular embedding output
    y_tp = tp_embedding(x.clone())
    y_raw = raw_embedding(x.clone())

    # Compare the results
    if rank == 0:
        logger.info(f"[Rank {rank}] Comparing results...")
        torch.testing.assert_close(y_tp, y_raw, atol=1e-5, rtol=1e-5)
        logger.info(f"[Rank {rank}] Test passed!")

def main():
    pcfg.setup_parallel_manager(config={"tp": 4})
    tensor_parallel_embedding_test(10000, 10)
    
    # Clean up the process group
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()