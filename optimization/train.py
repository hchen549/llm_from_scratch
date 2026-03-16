import math
import os
import time
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function, schedule

from .configs import DataConfig, ModelConfig, ProfilerConfig, TrainConfig
from .dataloader import create_dataloaders
from .model import SimpleMLP


class CudaH2DPrefetcher:
    """Prefetches batches onto GPU using a dedicated CUDA stream.

    While the default stream runs forward/backward on batch N,
    the transfer stream copies batch N+1 from pinned CPU memory to GPU.

        transfer_stream:  [H2D batch N+1] ───────────  [H2D batch N+2]
        default stream:     [fwd/bwd/opt batch N]        [fwd/bwd/opt batch N+1]
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        transfer_stream: torch.cuda.Stream,
    ):
        self.loader = loader
        self.device = device
        self.transfer_stream = transfer_stream

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        data_iter = iter(self.loader)

        # Prefetch the first batch on the transfer stream.
        try:
            batch = next(data_iter)
        except StopIteration:
            return

        with torch.cuda.stream(self.transfer_stream):
            next_x = batch[0].to(self.device, non_blocking=True)
            next_y = batch[1].to(self.device, non_blocking=True)

        for batch in data_iter:
            # Block default stream until the prefetched copy is done.
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            x, y = next_x, next_y

            # Start copying the next batch while compute runs on default stream.
            with torch.cuda.stream(self.transfer_stream):
                next_x = batch[0].to(self.device, non_blocking=True)
                next_y = batch[1].to(self.device, non_blocking=True)

            yield x, y

        # Yield the last prefetched batch.
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        yield next_x, next_y

    def __len__(self) -> int:
        return len(self.loader)


def get_cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_cfg: TrainConfig,
    global_step: int,
    warmup_steps: int,
    total_steps: int,
    prof: profile | None = None,
    transfer_stream: torch.cuda.Stream | None = None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Optionally wrap the loader with a prefetcher that copies batches on a
    # dedicated CUDA stream, overlapping H2D transfers with compute.
    batches = (
        CudaH2DPrefetcher(loader, device, transfer_stream)
        if train_cfg.use_prefetch and transfer_stream is not None
        else loader
    )

    for x, y in batches:
        if not (train_cfg.use_prefetch and transfer_stream is not None):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        # Update learning rate
        lr = get_cosine_lr(
            global_step, train_cfg.max_lr, train_cfg.min_lr, warmup_steps, total_steps
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with record_function("Forward"):
            logits = model(x)

        with record_function("Loss"):
            loss = criterion(logits, y)

        with record_function("Backward"):
            loss.backward()

        with record_function("Optimizer_Step"):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if prof is not None:
            prof.step()

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    data_cfg = DataConfig()
    profiler_cfg = ProfilerConfig()

    # Profiler output directory
    trace_dir = os.path.join(os.path.dirname(__file__), "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # --- Setup ---
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        num_samples=data_cfg.num_samples,
        in_features=model_cfg.in_features,
        num_classes=data_cfg.num_classes,
        batch_size=data_cfg.batch_size,
        val_fraction=data_cfg.val_fraction,
        num_workers=data_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleMLP(
        in_features=model_cfg.in_features,
        hidden_features=model_cfg.hidden_features,
        out_features=model_cfg.out_features,
        num_hidden_layers=model_cfg.num_hidden_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.max_lr, weight_decay=train_cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = train_cfg.num_epochs * len(train_loader)
    warmup_steps = int(total_steps * train_cfg.warmup_fraction)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")
    print(f"Prefetch: {train_cfg.use_prefetch}")
    print("-" * 60)

    # --- Profiler setup ---
    profiler_schedule = schedule(
        wait=profiler_cfg.wait,
        warmup=profiler_cfg.warmup,
        active=profiler_cfg.active,
        repeat=profiler_cfg.repeat,
    )

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_name = (
        "training_trace_prefetch.json"
        if train_cfg.use_prefetch
        else "training_trace_baseline.json"
    )
    trace_path: str = os.path.join(trace_dir, trace_name)

    def trace_handler(p):
        p.export_chrome_trace(trace_path)
        print(f"\n[Profiler] Chrome trace exported → {trace_path}")
        print("[Profiler] Open with: https://ui.perfetto.dev  or  chrome://tracing")

    # Create a single transfer stream for H2D prefetch (reused across all epochs).
    transfer_stream: torch.cuda.Stream | None = (
        torch.cuda.Stream(device=device)
        if train_cfg.use_prefetch and device.type == "cuda"
        else None
    )

    # --- Training loop ---
    global_step = 0
    start_time = time.time()

    with profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=trace_handler,
        record_shapes=profiler_cfg.record_shapes,
        profile_memory=profiler_cfg.profile_memory,
        with_stack=profiler_cfg.with_stack,
    ) as prof:
        for epoch in range(1, train_cfg.num_epochs + 1):
            train_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                train_cfg,
                global_step,
                warmup_steps,
                total_steps,
                prof=prof,
                transfer_stream=transfer_stream,
            )

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            current_lr = get_cosine_lr(
                global_step,
                train_cfg.max_lr,
                train_cfg.min_lr,
                warmup_steps,
                total_steps,
            )
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{train_cfg.num_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.2%} | "
                f"lr: {current_lr:.2e} | "
                f"time: {elapsed:.4f}s"
            )

    print("-" * 60)
    print("Training complete.")


if __name__ == "__main__":
    main()
