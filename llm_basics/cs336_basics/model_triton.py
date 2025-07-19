from ops.rms_norm import RmsNormTritonFunction

import torch 
import torch.nn as nn

class RMSNormTriton(nn.Module):
    def __init__(self, hidden_size, eps = 1e-5, device = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        return RmsNormTritonFunction.apply(x, self.weight, self.eps)

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"
