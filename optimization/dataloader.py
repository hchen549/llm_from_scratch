import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SyntheticClassificationDataset(Dataset):
    """Synthetic dataset for multi-class classification.

    Generates random input features and assigns labels based on a fixed
    random projection, producing a learnable (but noisy) classification task.
    """

    def __init__(self, num_samples: int = 10000, in_features: int = 784, num_classes: int = 10):
        super().__init__()
        self.data = torch.randn(num_samples, in_features)
        # Create a learnable mapping: project inputs and pick argmax as label
        projection = torch.randn(in_features, num_classes)
        logits = self.data @ projection
        self.targets = logits.argmax(dim=-1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def create_dataloaders(
    num_samples: int = 10000,
    in_features: int = 784,
    num_classes: int = 10,
    batch_size: int = 64,
    val_fraction: float = 0.2,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a synthetic dataset."""
    dataset = SyntheticClassificationDataset(num_samples, in_features, num_classes)

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
