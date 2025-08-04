import copy
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import argparse
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple
import statistics
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from .toy_model import ToyModel
from .dp_naive import DDP, DDPOverlapBucketed, DDPOverlapIndividual, DDPOverlapIndividual2
from llm_basics.cs336_basics.model import BasicsTransformerLM

# Configure logging to avoid issues with model initialization
logging.basicConfig(level=logging.WARNING)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters"""
    batch_size: int = 128
    vocab_size: int = 10000
    context_length: int = 512
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0
    num_epochs: int = 10
    warmup_epochs: int = 3
    bucket_size_mb: float = 1000.0


@dataclass
class TimingStats:
    """Statistics for timing measurements"""
    forward_time: List[float]
    backward_time: List[float]
    grad_sync_time: List[float]
    total_time: List[float]

    def __post_init__(self):
        if not hasattr(self, 'forward_time'):
            self.forward_time = []
        if not hasattr(self, 'backward_time'):
            self.backward_time = []
        if not hasattr(self, 'grad_sync_time'):
            self.grad_sync_time = []
        if not hasattr(self, 'total_time'):
            self.total_time = []

    def add_timing(self, forward: float, backward: float, grad_sync: float, total: float):
        """Add timing measurements for one iteration"""
        self.forward_time.append(forward)
        self.backward_time.append(backward)
        self.grad_sync_time.append(grad_sync)
        self.total_time.append(total)

    def get_averages(self) -> Dict[str, float]:
        """Calculate average times across all iterations"""
        return {
            'forward_avg': statistics.mean(self.forward_time) if self.forward_time else 0.0,
            'backward_avg': statistics.mean(self.backward_time) if self.backward_time else 0.0,
            'grad_sync_avg': statistics.mean(self.grad_sync_time) if self.grad_sync_time else 0.0,
            'total_avg': statistics.mean(self.total_time) if self.total_time else 0.0,
        }

    def get_communication_ratio(self) -> float:
        """Calculate the proportion of time spent on gradient communication"""
        if not self.total_time or not self.grad_sync_time:
            return 0.0
        return statistics.mean(self.grad_sync_time) / statistics.mean(self.total_time)


def create_transformer_model(config: BenchmarkConfig) -> BasicsTransformerLM:
    """Create a transformer model with the given configuration"""
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )


@contextmanager
def timer():
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def synchronize_cuda():
    """Synchronize CUDA operations for accurate timing"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_batch(batch_size: int, config: BenchmarkConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of token IDs for language model training"""
    # Use different seeds for each process to simulate real data
    torch.manual_seed(42 + dist.get_rank())
    
    # Generate random token IDs for input and targets
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
    # For language modeling, targets are input shifted by one position
    target_ids = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
    
    return input_ids, target_ids


def benchmark_ddp_implementation(
    ddp_class, 
    model: nn.Module, 
    config: BenchmarkConfig, 
    device: torch.device,
    **ddp_kwargs
) -> TimingStats:
    """Benchmark a specific DDP implementation"""
    
    try:
        # Create DDP model
        if ddp_class == DDPOverlapBucketed:
            if dist.get_rank() == 0:
                print(f"Creating DDPOverlapBucketed with bucket_size_mb={config.bucket_size_mb}")
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model has {total_params} trainable parameters")
            ddp_model = ddp_class(copy.deepcopy(model), bucket_size_mb=config.bucket_size_mb)
            if dist.get_rank() == 0:
                print(f"Successfully created DDPOverlapBucketed with {len(ddp_model.bucket_manager.buckets)} buckets")
        else:
            ddp_model = ddp_class(copy.deepcopy(model))
        
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        stats = TimingStats([], [], [], [])
        
        # Get micro batch size for this process
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        micro_batch_size = config.batch_size // world_size
        
        if rank == 0:
            print(f"Starting training loop with {config.num_epochs + config.warmup_epochs} total epochs")
        
        for epoch in range(config.num_epochs + config.warmup_epochs):
            try:
                input_ids, target_ids = get_batch(config.batch_size, config, device)
                
                # Shard data for this process
                start_idx = rank * micro_batch_size
                end_idx = (rank + 1) * micro_batch_size
                input_shard = input_ids[start_idx:end_idx]
                target_shard = target_ids[start_idx:end_idx]
                
                step_start = time.perf_counter()
                
                # Forward pass timing
                optimizer.zero_grad()
                synchronize_cuda()
                with timer() as forward_timer:
                    logits = ddp_model(input_shard)
                    # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), target_shard.view(-1))
                    synchronize_cuda()
                forward_time = forward_timer()
                
                # Backward pass timing
                synchronize_cuda()
                with timer() as backward_timer:
                    loss.backward()
                    synchronize_cuda()
                backward_time = backward_timer()
                
                # Gradient synchronization timing
                synchronize_cuda()
                with timer() as sync_timer:
                    ddp_model.finish_gradient_sync()
                    synchronize_cuda()
                grad_sync_time = sync_timer()
                
                optimizer.step()
                
                total_time = time.perf_counter() - step_start
                
                # Only record stats after warmup
                if epoch >= config.warmup_epochs:
                    stats.add_timing(forward_time, backward_time, grad_sync_time, total_time)
                    
            except Exception as e:
                if rank == 0:
                    print(f"Error in epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()
                raise e
        
        return stats
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error in benchmark_ddp_implementation for {ddp_class.__name__}: {e}")
            import traceback
            traceback.print_exc()
        raise e


def print_benchmark_results(implementation_name: str, stats: TimingStats, rank: int):
    """Print benchmark results for a specific implementation"""
    if rank != 0:
        return
    
    averages = stats.get_averages()
    comm_ratio = stats.get_communication_ratio()
    
    print(f"\n{'='*60}")
    print(f"Results for {implementation_name}")
    print(f"{'='*60}")
    print(f"Forward Pass Time:     {averages['forward_avg']*1000:.2f} ms")
    print(f"Backward Pass Time:    {averages['backward_avg']*1000:.2f} ms")
    print(f"Grad Sync Time:        {averages['grad_sync_avg']*1000:.2f} ms")
    print(f"Total Step Time:       {averages['total_avg']*1000:.2f} ms")
    print(f"Communication Ratio:   {comm_ratio*100:.1f}%")
    print(f"Throughput (steps/sec): {1.0/averages['total_avg']:.2f}")


def run_comprehensive_benchmark(config: BenchmarkConfig):
    """Run benchmark for all DDP implementations"""
    
    try:
        # Initialize distributed training
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Set device
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        
        if rank == 0:
            print(f"Starting benchmark with {world_size} processes")
            print(f"Configuration: {config}")
            print(f"Using device: {device}")
        
        # Create base model
        if rank == 0:
            print("Creating transformer model...")
        base_model = create_transformer_model(config).to(device)
        
        if rank == 0:
            total_params = sum(p.numel() for p in base_model.parameters())
            print(f"Model parameters: {total_params / 1e6:.2f}M")
            trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Benchmark each implementation
        implementations = [
            ("DDP (Synchronous)", DDP),
            ("DDP Individual Overlap", DDPOverlapIndividual),
            ("DDP Individual Overlap v2", DDPOverlapIndividual2),
            ("DDP Bucketed Overlap", DDPOverlapBucketed),
        ]
        
        all_results = {}
        
        for impl_name, impl_class in implementations:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Benchmarking {impl_name}...")
                print(f"{'='*60}")
            
            try:
                stats = benchmark_ddp_implementation(
                    impl_class, base_model, config, device
                )
                all_results[impl_name] = stats
                print_benchmark_results(impl_name, stats, rank)
                
            except Exception as e:
                if rank == 0:
                    import traceback
                    print(f"Error benchmarking {impl_name}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Full traceback:")
                    traceback.print_exc()
                    print(f"{'='*60}")
        
        # Print comparison summary
        if rank == 0 and all_results:
            print_comparison_summary(all_results, config)
            
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Fatal error in run_comprehensive_benchmark: {e}")
            import traceback
            traceback.print_exc()
        raise e


def print_comparison_summary(results: Dict[str, TimingStats], config: BenchmarkConfig):
    """Print a comparison summary of all implementations"""
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'Implementation':<25} {'Total (ms)':<12} {'Comm %':<8} {'Throughput':<12}")
    print("-" * 80)
    
    for impl_name, stats in results.items():
        averages = stats.get_averages()
        comm_ratio = stats.get_communication_ratio()
        throughput = 1.0 / averages['total_avg']
        
        print(f"{impl_name:<25} {averages['total_avg']*1000:<12.2f} "
              f"{comm_ratio*100:<8.1f} {throughput:<12.2f}")
    
    # Find best performing implementation
    best_impl = min(results.items(), key=lambda x: x[1].get_averages()['total_avg'])
    print(f"\nFastest Implementation: {best_impl[0]}")
    
    # Communication efficiency analysis
    print(f"\nCommunication Efficiency Analysis:")
    for impl_name, stats in results.items():
        comm_ratio = stats.get_communication_ratio()
        print(f"  {impl_name}: {comm_ratio*100:.1f}% time spent on communication")


def profile_ddp_implementation(
    ddp_class, 
    model: nn.Module, 
    config: BenchmarkConfig, 
    device: torch.device,
    output_dir: str,
    **ddp_kwargs
) -> None:
    """Profile a specific DDP implementation using PyTorch profiler"""
    
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if rank == 0:
            print(f"Profiling {ddp_class.__name__} implementation")
            print(f"Profile results will be saved to: {output_dir}")
        
        # Create DDP model
        if ddp_class == DDPOverlapBucketed:
            if rank == 0:
                print(f"Creating DDPOverlapBucketed with bucket_size_mb={config.bucket_size_mb}")
            ddp_model = ddp_class(copy.deepcopy(model), bucket_size_mb=config.bucket_size_mb)
        else:
            ddp_model = ddp_class(copy.deepcopy(model))
        
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        # Get micro batch size for this process
        micro_batch_size = config.batch_size // world_size
        
        # Define profiler activities
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        # Create profiler trace file name
        trace_file = os.path.join(output_dir, f"{ddp_class.__name__}_rank_{rank}_trace.json")
        
        total_epochs = config.num_epochs + config.warmup_epochs
        if rank == 0:
            print(f"Starting profiling with {config.warmup_epochs} wait steps + {config.num_epochs} training epochs")
        
        # Configure profiler schedule
        # wait: skip all warmup epochs plus initial steps
        # warmup: 0 (no profiler warmup needed)
        # active: only profile the regular training epochs
        # repeat: run once
        wait_steps = config.warmup_epochs  # Skip all warmup epochs
        warmup_steps = 0  # No profiler warmup
        active_steps = config.num_epochs  # Only profile regular epochs
        
        def trace_handler(prof):
            # Save trace when active profiling completes
            prof.export_chrome_trace(trace_file)
            if rank == 0:
                print(f"Saved profiler trace for regular training epochs (steps {prof.step_num - active_steps + 1} to {prof.step_num})")
        
        # Run profiling
        with profile(
            activities=activities,
            schedule=schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for epoch in range(total_epochs):
                # Determine epoch type for logging
                if epoch < config.warmup_epochs:
                    epoch_type = "warmup"  # Training warmup phase (profiler waiting)
                    epoch_num = epoch
                else:
                    epoch_type = "active"  # Regular training phase (profiler active)
                    epoch_num = epoch - config.warmup_epochs
                
                with record_function(f"{epoch_type}_epoch_{epoch_num}"):
                    try:
                        input_ids, target_ids = get_batch(config.batch_size, config, device)
                        
                        # Shard data for this process
                        start_idx = rank * micro_batch_size
                        end_idx = (rank + 1) * micro_batch_size
                        input_shard = input_ids[start_idx:end_idx]
                        target_shard = target_ids[start_idx:end_idx]
                        
                        with record_function("zero_grad"):
                            optimizer.zero_grad()
                        
                        # Forward pass
                        with record_function("forward_pass"):
                            logits = ddp_model(input_shard)
                            loss = loss_fn(logits.view(-1, logits.size(-1)), target_shard.view(-1))
                        
                        # Backward pass
                        with record_function("backward_pass"):
                            loss.backward()
                        
                        # Gradient synchronization
                        with record_function("gradient_sync"):
                            ddp_model.finish_gradient_sync()
                        
                        with record_function("optimizer_step"):
                            optimizer.step()
                            
                        # Add profiler step for timeline visualization
                        prof.step()
                        
                    except Exception as e:
                        if rank == 0:
                            print(f"Error in profiling epoch {epoch}: {e}")
                        raise e
        
        if rank == 0:
            print(f"Profiling completed. Trace saved to: {trace_file}")
            print(f"You can view the trace using:")
            print(f"  chrome://tracing/ (paste the .json file)")
            print(f"  or use tensorboard: tensorboard --logdir {output_dir}")
            
            # Save a summary report
            summary_file = os.path.join(output_dir, f"{ddp_class.__name__}_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Profiling Summary for {ddp_class.__name__}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Model Configuration:\n")
                f.write(f"  Batch size: {config.batch_size}\n")
                f.write(f"  Vocabulary size: {config.vocab_size}\n")
                f.write(f"  Context length: {config.context_length}\n")
                f.write(f"  Model dimension: {config.d_model}\n")
                f.write(f"  Number of layers: {config.num_layers}\n")
                f.write(f"  Number of heads: {config.num_heads}\n")
                f.write(f"  Feed-forward dimension: {config.d_ff}\n")
                f.write(f"  Warmup epochs: {config.warmup_epochs}\n")
                f.write(f"  Training epochs: {config.num_epochs}\n")
                f.write(f"  Total epochs: {total_epochs}\n")
                f.write(f"  Profiler wait steps: {wait_steps}\n")
                f.write(f"  Profiler warmup steps: {warmup_steps}\n")
                f.write(f"  Profiler active steps: {active_steps}\n")
                f.write(f"  World size: {world_size}\n")
                f.write(f"  Device: {device}\n")
                f.write(f"\nTrace file: {trace_file}\n")
            
            print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error in profile_ddp_implementation for {ddp_class.__name__}: {e}")
            import traceback
            traceback.print_exc()
        raise e


def run_profiling_benchmark(config: BenchmarkConfig, implementation: str, output_dir: str):
    """Run profiling for a specific DDP implementation"""
    
    try:
        # Initialize distributed training
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Set device
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        
        if rank == 0:
            print(f"Starting profiling with {world_size} processes")
            print(f"Configuration: {config}")
            print(f"Using device: {device}")
            print("=" * 60)
            print("PYTORCH PROFILER MODE")
            print("=" * 60)
            print(f"Implementation: {implementation}")
            print(f"Output directory: {output_dir}")
            print(f"Warmup epochs: {config.warmup_epochs}")
            print(f"Training epochs: {config.num_epochs}")
            print("=" * 60)
        
        # Create base model
        if rank == 0:
            print("Creating transformer model...")
        base_model = create_transformer_model(config).to(device)
        
        if rank == 0:
            total_params = sum(p.numel() for p in base_model.parameters())
            print(f"Model parameters: {total_params / 1e6:.2f}M")
        
        # Map implementation names to classes
        implementations = {
            "ddp": DDP,
            "ddp_individual": DDPOverlapIndividual,
            "ddp_individual_v2": DDPOverlapIndividual2,
            "ddp_bucketed": DDPOverlapBucketed,
        }
        
        if implementation not in implementations:
            if rank == 0:
                print(f"Error: Unknown implementation '{implementation}'")
                print(f"Available implementations: {list(implementations.keys())}")
            return
        
        impl_class = implementations[implementation]
        impl_name = {
            "ddp": "DDP (Synchronous)",
            "ddp_individual": "DDP Individual Overlap", 
            "ddp_individual_v2": "DDP Individual Overlap v2",
            "ddp_bucketed": "DDP Bucketed Overlap"
        }[implementation]
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Profiling {impl_name}...")
            print(f"{'='*60}")
        
        profile_ddp_implementation(
            impl_class, base_model, config, device, output_dir
        )
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Fatal error in run_profiling_benchmark: {e}")
            import traceback
            traceback.print_exc()
        raise e


def main():
    """
    Benchmark or profile DDP implementations with Transformer model.
    
    This script supports two modes:
    1. Benchmark mode (default): Compares all DDP implementations and shows performance metrics
    2. Profile mode (--profile): Uses PyTorch profiler to create detailed CPU/GPU activity traces
    
    Examples:
        # Run comprehensive benchmark
        python benchmark_ddp.py --batch-size 64 --num-layers 6
        
        # Profile a specific DDP implementation
        python benchmark_ddp.py --profile --profile-implementation ddp_bucketed --profile-output-dir ./traces
        
        # Profile with custom model configuration
        python benchmark_ddp.py --profile --profile-implementation ddp_individual_v2 \\
                                --d-model 512 --num-layers 8 --profile-epochs 3
    """
    parser = argparse.ArgumentParser(description="Benchmark DDP implementations with Transformer model")
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta value")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--bucket-size-mb", type=float, default=100.0, help="Bucket size in MB")
    
    # Profiling arguments
    parser.add_argument("--profile", action="store_true", 
                       help="Enable profiling mode to generate detailed CPU/GPU activity traces")
    parser.add_argument("--profile-implementation", type=str, 
                       choices=["ddp", "ddp_individual", "ddp_individual_v2", "ddp_bucketed"],
                       default="ddp", help="DDP implementation to profile")
    parser.add_argument("--profile-output-dir", type=str, default="./profiler_output",
                       help="Directory to save profiler results (JSON traces and summary)")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        bucket_size_mb=args.bucket_size_mb
    )
    
    if args.profile:
        run_profiling_benchmark(
            config, 
            args.profile_implementation, 
            args.profile_output_dir
        )
    else:
        run_comprehensive_benchmark(config)


if __name__ == "__main__":
    main() 