from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

import timeit
import statistics
import torch
import random
import logging
from torch.profiler import profile, record_function, ProfilerActivity
import os
import click
import sys
from pathlib import Path

from llm_basics.cs336_basics.model_patch import apply_triton_kernel

# Add parent directory to path to import config modules
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config_dataclasses import load_config_from_file
except ImportError:
    load_config_from_file = None




def create_dummy_input_variable_length(context_length, batch_size, vocab_size):
    batch_inputs = []
    for _ in range(batch_size):
        seq_len = random.randint(1, context_length)
        seq = torch.randint(0, vocab_size, (seq_len,))
        batch_inputs.append(seq)
    
    max_len = max(len(seq) for seq in batch_inputs)
    padded_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, seq in enumerate(batch_inputs):
        padded_batch[i, :len(seq)] = seq
    
    return padded_batch

def create_dummy_input_fixed_length(seq_len, batch_size, vocab_size):
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    return dummy_input

def run_benchmark(cfg) -> None:
    """
    Main function that takes a configuration object.
    This is the cleaner version of the main function.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set random seed
    random.seed(cfg.system.random_seed)
    
    # Log configuration
    if hasattr(cfg, 'log_configuration'):
        cfg.log_configuration(logging)
    else:
        # Fallback logging
        logging.info("Configuration:")
        logging.info(f"  Model: vocab_size={cfg.model.vocab_size}, context_length={cfg.model.context_length}, d_model={cfg.model.d_model}")
        logging.info(f"  Model: num_layers={cfg.model.num_layers}, num_heads={cfg.model.num_heads}, d_ff={cfg.model.d_ff}, rope_theta={cfg.model.rope_theta}")
        logging.info(f"  Benchmark: batch_size={cfg.benchmark.batch_size}, forward_and_backward={cfg.benchmark.forward_and_backward_pass}")
        logging.info(f"  Benchmark: warmup_iterations={cfg.benchmark.warmup_iterations}, num_iterations={cfg.benchmark.num_iterations}")
        logging.info(f"  Benchmark: variable_length={cfg.benchmark.variable_length}")
        logging.info(f"  Optimizer: learning_rate={cfg.optimizer.learning_rate}, foreach={cfg.optimizer.foreach}")
        logging.info(f"  Profiler: enable={cfg.profiler.enable_profiler}, output_dir={cfg.profiler.profiler_output_dir}")
        logging.info(f"  System: random_seed={cfg.system.random_seed}")
    
    # Create profiler output directory if enabled
    if cfg.profiler.enable_profiler:
        os.makedirs(cfg.profiler.profiler_output_dir, exist_ok=True)
    
    model = BasicsTransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    )
    model.to("cuda")
    apply_triton_kernel(model, rms = True)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, foreach=cfg.optimizer.foreach)
    
    dummy_inputs = []
    for _ in range(cfg.benchmark.warmup_iterations + cfg.benchmark.num_iterations):
        if cfg.benchmark.variable_length:
            dummy_inputs.append(create_dummy_input_variable_length(cfg.model.context_length, cfg.benchmark.batch_size, cfg.model.vocab_size).to("cuda"))
        else:
            dummy_inputs.append(create_dummy_input_fixed_length(cfg.model.context_length, cfg.benchmark.batch_size, cfg.model.vocab_size).to("cuda"))
    
    iteration_idx = 0

    # Measure time
    def forward_pass(input_data):
        with record_function("forward_pass"):
            outputs = model(input_data)
            loss = cross_entropy(outputs, input_data)
            torch.cuda.synchronize()
        return loss

    def backward_pass(loss):
        with record_function("backward_pass"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

    def one_pass(input_data):
        loss = forward_pass(input_data)
        if cfg.benchmark.forward_and_backward_pass:
            backward_pass(loss)

    # warmup
    for _ in range(cfg.benchmark.warmup_iterations):
        if cfg.benchmark.forward_and_backward_pass:
            one_pass(dummy_inputs[iteration_idx])
        else:
            forward_pass(dummy_inputs[iteration_idx])
        iteration_idx += 1

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Create profiler context manager
    if cfg.profiler.enable_profiler:
        profiler_cm = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    else:
        from contextlib import nullcontext
        profiler_cm = nullcontext()

    # Measure time for forward and backward pass
    times = []
    with profiler_cm as prof:
        for i in range(cfg.benchmark.num_iterations):
            start_time = timeit.default_timer()
            with record_function(f"iteration_{i}"):
                if cfg.benchmark.forward_and_backward_pass:
                    one_pass(dummy_inputs[iteration_idx])
                else:
                    forward_pass(dummy_inputs[iteration_idx])
            end_time = timeit.default_timer()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            iteration_idx += 1
            
            # Step profiler if enabled
            if cfg.profiler.enable_profiler:
                prof.step()

    if cfg.profiler.enable_profiler:
        # Save additional profiler trace formats
        trace_path = os.path.join(cfg.profiler.profiler_output_dir, cfg.profiler.profiler_trace_file)
        memory_timeline_path = os.path.join(cfg.profiler.profiler_output_dir, cfg.profiler.memory_timeline_file)
        tensorboard_path = os.path.join(cfg.profiler.profiler_output_dir, "tensorboard_trace.json")
        
        # Export trace as JSON for Chrome
        prof.export_chrome_trace(trace_path)
        logging.info(f"Chrome trace saved to: {trace_path}")
        
        # Export TensorBoard trace
        try:
            prof.export_stacks(tensorboard_path, "self_cuda_time_total")
            logging.info(f"TensorBoard trace saved to: {tensorboard_path}")
        except Exception as e:
            logging.warning(f"Could not save TensorBoard trace: {e}")
        
        # Export memory timeline as HTML
        try:
            prof.export_memory_timeline(memory_timeline_path)
            logging.info(f"Memory timeline saved to: {memory_timeline_path}")
        except Exception as e:
            logging.warning(f"Could not save memory timeline: {e}")

        # Save memory snapshot with custom filename
        memory_snapshot_file = os.path.join(cfg.profiler.profiler_output_dir, "memory_snapshot.pickle") 
        torch.cuda.memory._dump_snapshot(memory_snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None)
        logging.info(f"Memory snapshot saved to: {memory_snapshot_file}")

    # Calculate average time
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times)
    print(f"Average time per iteration: {avg_time:.6f} seconds")
    print(f"Standard deviation: {std_time:.6f} seconds")


@click.command()
@click.option("--config", "-c", required=True, help="Path to YAML configuration file")
def main(config):
    """Run the benchmark with the specified configuration file."""
    if load_config_from_file is None:
        click.echo("Error: Could not import config_dataclasses. Make sure it's available.", err=True)
        sys.exit(1)
    
    # Load configuration from file
    cfg = load_config_from_file(config)
    
    # Run the benchmark
    run_benchmark(cfg)

if __name__ == "__main__":
    main()