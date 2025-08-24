from dataclasses import dataclass
from transformers import AutoTokenizer
from example.llama.load_hf_direct_rope import load_model
from typing import List, Dict

import torch
import time
import statistics
import numpy as np

@dataclass
class InputTokensConfig:
    batch_size: int = 4
    input_len: int = 256

@dataclass
class ModelGenerationConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    model_type: str = "llama3_hf_rope_fast_inference"
    attention_type: str = "paged"
    output_len: int = 100
    temperature: float = 0.5
    top_p: float = 0.9
    eos_token_id: int = None
    warm_up_steps: int = 10
    num_steps: int = 10

def generate_input_tokens(tokenizer: AutoTokenizer, input_tokens_config: InputTokensConfig):
    return torch.randint(0, tokenizer.vocab_size, (input_tokens_config.batch_size, input_tokens_config.input_len))

def benchmark_kvcache(generate_config: ModelGenerationConfig, input_tokens_config: InputTokensConfig) -> Dict[str, float]:
    """
    Benchmark KV cache model generation and return throughput metrics.
    
    Returns:
        Dictionary containing throughput statistics
    """
    print(f"Loading model: {generate_config.model_name}")
    hf_model, model = load_model(generate_config.model_name, model_type=generate_config.model_type, attention_type=generate_config.attention_type)
    tokenizer = AutoTokenizer.from_pretrained(generate_config.model_name)
    
    # Calculate tokens generated per step
    tokens_per_step = generate_config.output_len * input_tokens_config.batch_size
    
    print(f"\nBenchmark Configuration:")
    print(f"  Batch size: {input_tokens_config.batch_size}")
    print(f"  Input sequence length: {input_tokens_config.input_len}")
    print(f"  Max new tokens per sequence: {generate_config.output_len}")
    print(f"  Total tokens generated per step: {tokens_per_step}")
    print(f"  Warm-up steps: {generate_config.warm_up_steps}")
    print(f"  Benchmark steps: {generate_config.num_steps}")
    print(f"  Temperature: {generate_config.temperature}, Top-p: {generate_config.top_p}")
    
    # Warm up
    print(f"\nWarming up for {generate_config.warm_up_steps} steps...")
    for i in range(generate_config.warm_up_steps):
        input_tokens = generate_input_tokens(tokenizer, input_tokens_config).cuda()
        _ = model.generate(
            input_tokens, 
            generate_config.output_len, 
            generate_config.temperature, 
            generate_config.top_p, 
            eos_token_id=generate_config.eos_token_id
        )
    
    # Benchmark
    print(f"\nRunning benchmark for {generate_config.num_steps} steps...")
    step_times: List[float] = []
    step_throughputs: List[float] = []
    
    total_start_time = time.time()
    
    for i in range(generate_config.num_steps):
        input_tokens = generate_input_tokens(tokenizer, input_tokens_config).cuda()
        
        step_start_time = time.time()
        output_tokens = model.generate(
            input_tokens, 
            generate_config.output_len, 
            generate_config.temperature, 
            generate_config.top_p, 
            eos_token_id=generate_config.eos_token_id
        )
        step_end_time = time.time()
        
        step_time = step_end_time - step_start_time
        step_throughput = tokens_per_step / step_time
        
        step_times.append(step_time)
        step_throughputs.append(step_throughput)
        
        print(f"  Step {i+1:2d}: {step_time:.3f}s, {step_throughput:.1f} tokens/sec")
    
    total_end_time = time.time()
    
    # Calculate statistics
    total_time = total_end_time - total_start_time
    total_tokens = tokens_per_step * generate_config.num_steps
    overall_throughput = total_tokens / total_time
    
    avg_step_time = statistics.mean(step_times)
    avg_throughput = statistics.mean(step_throughputs)
    min_throughput = min(step_throughputs)
    max_throughput = max(step_throughputs)
    std_throughput = statistics.stdev(step_throughputs) if len(step_throughputs) > 1 else 0.0
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print(f"THROUGHPUT BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total tokens generated: {total_tokens:,}")
    print(f"Total benchmark time: {total_time:.3f}s")
    print(f"Overall throughput: {overall_throughput:.1f} tokens/sec")
    print(f"\nPer-step Statistics:")
    print(f"  Average step time: {avg_step_time:.3f}s")
    print(f"  Average throughput: {avg_throughput:.1f} tokens/sec")
    print(f"  Min throughput: {min_throughput:.1f} tokens/sec")
    print(f"  Max throughput: {max_throughput:.1f} tokens/sec")
    print(f"  Std deviation: {std_throughput:.1f} tokens/sec")
    print(f"  Coefficient of variation: {(std_throughput/avg_throughput)*100:.1f}%")
    
    # Calculate percentiles
    p50 = np.percentile(step_throughputs, 50)
    p90 = np.percentile(step_throughputs, 90)
    p99 = np.percentile(step_throughputs, 99)
    
    print(f"\nThroughput Percentiles:")
    print(f"  P50 (median): {p50:.1f} tokens/sec")
    print(f"  P90: {p90:.1f} tokens/sec")
    print(f"  P99: {p99:.1f} tokens/sec")
    
    # Return metrics for programmatic use
    return {
        'overall_throughput': overall_throughput,
        'avg_throughput': avg_throughput,
        'min_throughput': min_throughput,
        'max_throughput': max_throughput,
        'std_throughput': std_throughput,
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_step_time': avg_step_time,
        'p50_throughput': p50,
        'p90_throughput': p90,
        'p99_throughput': p99
    } 

if __name__ == "__main__":
    # Default configurations
    generate_config = ModelGenerationConfig()
    input_tokens_config = InputTokensConfig()
    
    print("Starting KV Cache Throughput Benchmark")
    print("=" * 60)
    
    # Run the benchmark
    metrics = benchmark_kvcache(generate_config, input_tokens_config)
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Key Metrics:")
    print(f"  Overall Throughput: {metrics['overall_throughput']:.1f} tokens/sec")
    print(f"  Average Throughput: {metrics['avg_throughput']:.1f} tokens/sec")
    print(f"  P50 Throughput: {metrics['p50_throughput']:.1f} tokens/sec")
    print(f"  Throughput Range: {metrics['min_throughput']:.1f} - {metrics['max_throughput']:.1f} tokens/sec")