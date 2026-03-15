import math
import time

import torch
import torch.nn as nn

from .dataloader import create_dataloaders
from .model import SimpleMLP


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
    epoch: int,
    global_step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Update learning rate
        lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_steps, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

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
    # --- Hyperparameters ---
    seed = 42
    num_epochs = 40
    batch_size = 128
    max_lr = 3e-3
    min_lr = 1e-5
    weight_decay = 0.01
    warmup_fraction = 0.1

    # Data config
    num_samples = 10000
    in_features = 784
    num_classes = 10

    # --- Setup ---
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        num_samples=num_samples,
        in_features=in_features,
        num_classes=num_classes,
        batch_size=batch_size,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleMLP(
        in_features=in_features,
        hidden_features=256,
        out_features=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_fraction)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")
    print("-" * 60)

    # --- Training loop ---
    global_step = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            global_step,
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = get_cosine_lr(
            global_step, max_lr, min_lr, warmup_steps, total_steps
        )
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.2%} | "
            f"lr: {current_lr:.2e} | "
            f"time: {elapsed:.1f}s"
        )

    print("-" * 60)
    print("Training complete.")


if __name__ == "__main__":
    main()
