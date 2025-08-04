# DDP Profiling with PyTorch Profiler

This document explains how to use the PyTorch profiler functionality added to the DDP benchmark script.

## Overview

The benchmark script now supports two modes:

1. **Benchmark mode** (default): Compares all DDP implementations and shows performance metrics
2. **Profile mode** (`--profile`): Uses PyTorch profiler to create detailed CPU/GPU activity traces for a specific DDP implementation

## Profiling Mode Usage

### Basic Profiling

```bash
# Profile the basic DDP implementation
python benchmark_ddp.py --profile

# Profile a specific implementation
python benchmark_ddp.py --profile --profile-implementation ddp_bucketed

# Specify output directory
python benchmark_ddp.py --profile --profile-output-dir ./my_traces
```

### Available DDP Implementations

- `ddp`: Basic synchronous DDP
- `ddp_individual`: Individual parameter overlap
- `ddp_individual_v2`: Individual parameter overlap v2
- `ddp_bucketed`: Bucketed gradient overlap

### Custom Model Configuration

```bash
# Profile with smaller model for faster iteration
python benchmark_ddp.py --profile \
    --profile-implementation ddp_individual_v2 \
    --d-model 512 \
    --num-layers 8 \
    --batch-size 16 \
    --num-epochs 3 \
    --warmup-epochs 1

# Profile with larger model
python benchmark_ddp.py --profile \
    --profile-implementation ddp_bucketed \
    --d-model 1024 \
    --num-layers 24 \
    --batch-size 64 \
    --bucket-size-mb 1000 \
    --num-epochs 5 \
    --warmup-epochs 2
```

## Profiler Output

### Files Generated

1. **Trace files**: `{implementation}_rank_{rank}_trace.json`
   - Chrome trace format files for each rank
   - Can be viewed in Chrome DevTools (`chrome://tracing/`)

2. **Summary file**: `{implementation}_summary.txt`
   - Text summary of profiling configuration and results

### Viewing Traces

#### Option 1: Chrome DevTools
1. Open Chrome browser
2. Go to `chrome://tracing/`
3. Click "Load" and select the `.json` trace file
4. Explore timeline view with CPU/GPU activities

#### Option 2: TensorBoard (if available)
```bash
tensorboard --logdir ./profiler_output
```

### Understanding the Trace

The profiler records:
- **CPU activities**: Function calls, data loading, gradient computation
- **GPU activities**: CUDA kernels, memory transfers
- **Memory usage**: Memory allocations and deallocations
- **Communication**: All-reduce operations and gradient synchronization
- **Learning rate scheduling**: LR updates during warmup phases
- **Gradient clipping**: Applied only during warmup epochs

Key sections to look for:
- `forward_pass`: Model forward computation
- `backward_pass`: Gradient computation
- `gradient_sync`: DDP communication (all-reduce)
- `optimizer_step`: Parameter updates
- `lr_scheduling`: Learning rate updates (warmup phases)
- `gradient_clipping_warmup`: Gradient norm clipping during warmup

## Warmup Scheduling Features

The profiler now includes comprehensive warmup scheduling:

### 1. **Learning Rate Warmup**
- Linear warmup from 0 to base learning rate (0.001) over warmup epochs
- Constant learning rate after warmup completion
- Each epoch trace includes current learning rate in the name

### 2. **Gradient Clipping During Warmup**
- Gradient norm clipping (max_norm=1.0) applied only during warmup epochs
- Helps stabilize training during the initial phase
- Clearly visible in profiler traces

### 3. **Progress Logging**
- Periodic logging of epoch type, learning rate, and loss
- Easy identification of warmup vs training phases
- Performance tracking across the entire training process

## Example Workflow

1. **Start with basic profiling**:
   ```bash
   python benchmark_ddp.py --profile --num-epochs 3 --warmup-epochs 1
   ```

2. **Compare implementations**:
   ```bash
   # Profile synchronous DDP
   python benchmark_ddp.py --profile --profile-implementation ddp --profile-output-dir ./traces/ddp

   # Profile bucketed overlap
   python benchmark_ddp.py --profile --profile-implementation ddp_bucketed --profile-output-dir ./traces/ddp_bucketed
   ```

3. **Analyze results**:
   - Load traces in Chrome DevTools
   - Compare communication patterns
   - Identify bottlenecks and overlap efficiency

## Tips for Profiling

- Use fewer epochs (`--num-epochs 3-5`) for detailed tracing
- Use minimal warmup (`--warmup-epochs 1-2`) to focus on steady-state performance
- Start with smaller models for faster iteration
- Profile on actual multi-GPU setup to see real communication patterns
- Focus on the gradient synchronization sections to understand DDP differences
- Look for differences between warmup and training epochs in the trace

## Running on Multiple GPUs

```bash
# Example with torchrun
torchrun --nproc_per_node=4 benchmark_ddp.py --profile --profile-implementation ddp_bucketed
```

This will generate trace files for each rank, allowing you to see the full distributed training timeline. 