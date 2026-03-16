from dataclasses import dataclass


@dataclass
class ModelConfig:
    in_features: int = 8192
    hidden_features: int = 2048
    out_features: int = 10
    num_hidden_layers: int = 4


@dataclass
class DataConfig:
    num_samples: int = 10000
    num_classes: int = 10
    batch_size: int = 512
    val_fraction: float = 0.2
    num_workers: int = 2


@dataclass
class TrainConfig:
    seed: int = 42
    num_epochs: int = 20
    max_lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    use_prefetch: bool = False


@dataclass
class ProfilerConfig:
    enabled: bool = True
    wait: int = 2
    warmup: int = 3
    active: int = 25
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False
