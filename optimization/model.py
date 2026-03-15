import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """A simple multi-layer perceptron for classification."""

    def __init__(
        self,
        in_features: int = 784,
        hidden_features: int = 256,
        out_features: int = 10,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.BatchNorm1d(hidden_features))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
