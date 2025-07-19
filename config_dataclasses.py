#!/usr/bin/env python3
"""
Configuration dataclasses for CS336 Assignment 2 Systems Benchmark.

This module defines dataclasses that group related configuration parameters
with default values, making the code more organized and maintainable.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture parameters."""
    vocab_size: int = 100000
    context_length: int = 256
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution parameters."""
    batch_size: int = 4
    forward_and_backward_pass: bool = True
    warmup_iterations: int = 10
    num_iterations: int = 10
    variable_length: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizer parameters."""
    learning_rate: float = 1e-4
    foreach: bool = True


@dataclass
class ProfilerConfig:
    """Configuration for profiler settings."""
    enable_profiler: bool = False
    profiler_output_dir: str = "./profiler_traces"
    profiler_trace_file: str = "trace.json"
    memory_timeline_file: str = "memory_timeline.html"


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    device: str = "cuda"
    random_seed: int = 42


@dataclass
class TritonConfig:
    """Configuration for Triton kernel settings."""
    rms: bool = False


@dataclass
class BenchmarkConfiguration:
    """Complete benchmark configuration containing all parameter groups."""
    model: ModelConfig
    benchmark: BenchmarkConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig
    system: SystemConfig
    triton: TritonConfig

    @classmethod
    def from_defaults(cls) -> "BenchmarkConfiguration":
        """Create a configuration with all default values."""
        return cls(
            model=ModelConfig(),
            benchmark=BenchmarkConfig(),
            optimizer=OptimizerConfig(),
            profiler=ProfilerConfig(),
            system=SystemConfig(),
            triton=TritonConfig()
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BenchmarkConfiguration":
        """Create a configuration from a dictionary (e.g., loaded from YAML)."""
        def get_or_default(config_dict: dict, key: str, default_class):
            if key in config_dict and isinstance(config_dict[key], dict):
                # Filter out None values and unknown keys
                valid_fields = {field.name for field in default_class.__dataclass_fields__.values()}
                filtered_dict = {k: v for k, v in config_dict[key].items() 
                               if v is not None and k in valid_fields}
                return default_class(**filtered_dict)
            else:
                return default_class()

        return cls(
            model=get_or_default(config_dict, 'model', ModelConfig),
            benchmark=get_or_default(config_dict, 'benchmark', BenchmarkConfig),
            optimizer=get_or_default(config_dict, 'optimizer', OptimizerConfig),
            profiler=get_or_default(config_dict, 'profiler', ProfilerConfig),
            system=get_or_default(config_dict, 'system', SystemConfig),
            triton=get_or_default(config_dict, 'triton', TritonConfig)
        )

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            'model': {
                'vocab_size': self.model.vocab_size,
                'context_length': self.model.context_length,
                'd_model': self.model.d_model,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'd_ff': self.model.d_ff,
                'rope_theta': self.model.rope_theta,
            },
            'benchmark': {
                'batch_size': self.benchmark.batch_size,
                'forward_and_backward_pass': self.benchmark.forward_and_backward_pass,
                'warmup_iterations': self.benchmark.warmup_iterations,
                'num_iterations': self.benchmark.num_iterations,
                'variable_length': self.benchmark.variable_length,
            },
            'optimizer': {
                'learning_rate': self.optimizer.learning_rate,
                'foreach': self.optimizer.foreach,
            },
            'profiler': {
                'enable_profiler': self.profiler.enable_profiler,
                'profiler_output_dir': self.profiler.profiler_output_dir,
                'profiler_trace_file': self.profiler.profiler_trace_file,
                'memory_timeline_file': self.profiler.memory_timeline_file,
            },
            'system': {
                'device': self.system.device,
                'random_seed': self.system.random_seed,
            },
            'triton': {
                'rms': self.triton.rms,
            }
        }

    def apply_cli_overrides(self, **kwargs) -> None:
        """Apply command-line argument overrides to the configuration."""
        # Model overrides
        if kwargs.get('context_length') is not None:
            self.model.context_length = kwargs['context_length']
        if kwargs.get('d_model') is not None:
            self.model.d_model = kwargs['d_model']
        if kwargs.get('num_layers') is not None:
            self.model.num_layers = kwargs['num_layers']
        if kwargs.get('num_heads') is not None:
            self.model.num_heads = kwargs['num_heads']
        if kwargs.get('d_ff') is not None:
            self.model.d_ff = kwargs['d_ff']
        if kwargs.get('rope_theta') is not None:
            self.model.rope_theta = kwargs['rope_theta']

        # Benchmark overrides
        if kwargs.get('forward_and_backward_pass') is not None:
            self.benchmark.forward_and_backward_pass = kwargs['forward_and_backward_pass']
        if kwargs.get('warmup_iterations') is not None:
            self.benchmark.warmup_iterations = kwargs['warmup_iterations']
        if kwargs.get('num_iterations') is not None:
            self.benchmark.num_iterations = kwargs['num_iterations']
        if kwargs.get('variable_length') is not None:
            self.benchmark.variable_length = kwargs['variable_length']

        # Profiler overrides
        if kwargs.get('enable_profiler') is not None:
            self.profiler.enable_profiler = kwargs['enable_profiler']
        if kwargs.get('profiler_output_dir') is not None:
            self.profiler.profiler_output_dir = kwargs['profiler_output_dir']
        if kwargs.get('profiler_trace_file') is not None:
            self.profiler.profiler_trace_file = kwargs['profiler_trace_file']
        if kwargs.get('memory_timeline_file') is not None:
            self.profiler.memory_timeline_file = kwargs['memory_timeline_file']

    def log_configuration(self, logger) -> None:
        """Log the current configuration using the provided logger."""
        logger.info("Configuration:")
        logger.info(f"  Model: vocab_size={self.model.vocab_size}, context_length={self.model.context_length}, d_model={self.model.d_model}")
        logger.info(f"  Model: num_layers={self.model.num_layers}, num_heads={self.model.num_heads}, d_ff={self.model.d_ff}, rope_theta={self.model.rope_theta}")
        logger.info(f"  Benchmark: batch_size={self.benchmark.batch_size}, forward_and_backward={self.benchmark.forward_and_backward_pass}")
        logger.info(f"  Benchmark: warmup_iterations={self.benchmark.warmup_iterations}, num_iterations={self.benchmark.num_iterations}")
        logger.info(f"  Benchmark: variable_length={self.benchmark.variable_length}")
        logger.info(f"  Optimizer: learning_rate={self.optimizer.learning_rate}, foreach={self.optimizer.foreach}")
        logger.info(f"  Profiler: enable={self.profiler.enable_profiler}, output_dir={self.profiler.profiler_output_dir}")
        logger.info(f"  System: random_seed={self.system.random_seed}")
        logger.info(f"  Triton: rms={self.triton.rms}")


def load_config_from_file(config_path: Optional[str] = None) -> BenchmarkConfiguration:
    """
    Load configuration from a YAML file or create default configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        BenchmarkConfiguration instance
    """
    if not config_path:
        return BenchmarkConfiguration.from_defaults()
    
    try:
        from pathlib import Path
        import yaml
        
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found, using defaults")
            return BenchmarkConfiguration.from_defaults()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        
        print(f"Loaded configuration from: {config_path}")
        return BenchmarkConfiguration.from_dict(config_dict)
        
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        print("Using default configuration")
        return BenchmarkConfiguration.from_defaults()


def create_config_from_cli_args(**kwargs) -> BenchmarkConfiguration:
    """
    Create a configuration by loading from file and applying CLI overrides.
    
    Args:
        **kwargs: Command-line arguments including 'config' for file path
        
    Returns:
        BenchmarkConfiguration instance with CLI overrides applied
    """
    config_path = kwargs.get('config')
    config = load_config_from_file(config_path)
    
    # Apply CLI overrides
    config.apply_cli_overrides(**kwargs)
    
    return config