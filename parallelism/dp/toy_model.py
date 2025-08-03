import torch
import torch.nn as nn

class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class ToyModel(nn.Module):
    def __init__(self, in_features: int = 10, out_features: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, out_features, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(
            torch.tensor([2.0, 2.0]), requires_grad=False
        )

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x