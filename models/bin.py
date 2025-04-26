import torch
from torch import nn

#reference: https://github.com/LeonardoBerti00/TLOB/blob/main/models/bin.py
# This is a PyTorch implementation of the BiN (Batch Instance Normalization) layer.
# The BiN layer is a combination of batch normalization and instance normalization.
# It normalizes the input tensor along both the batch and feature dimensions.
# The BiN layer is designed to work with 3D tensors, typically used in time series or sequential data.

class BiN(nn.Module):
    def __init__(self, d1, t1):
        super().__init__()
        self.d1 = d1  # feature dimension
        self.t1 = t1  # time dimension

        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)

        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)

        self.y1 = nn.Parameter(torch.tensor(0.5))
        self.y2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Clamp y1 and y2 to stay positive
        self.y1.data.clamp_(min=0.01)
        self.y2.data.clamp_(min=0.01)

        # --- Temporal normalization (normalize across time dimension: dim=2)
        mean_time = x.mean(dim=2, keepdim=True)          # (batch, feature, 1)
        std_time = x.std(dim=2, keepdim=True)
        std_time = torch.where(std_time < 1e-4, torch.ones_like(std_time), std_time)

        z_time = (x - mean_time) / std_time               # (batch, feature, time)

        # Apply learnable scale and bias
        l2 = self.l2.transpose(0, 1)                      # (1, feature)
        B2 = self.B2.transpose(0, 1)                      # (1, feature)
        X2 = l2.unsqueeze(-1) * z_time + B2.unsqueeze(-1)  # (batch, feature, time)

        # --- Feature normalization (normalize across feature dimension: dim=1)
        mean_feat = x.mean(dim=1, keepdim=True)           # (batch, 1, time)
        std_feat = x.std(dim=1, keepdim=True)
        std_feat = torch.where(std_feat < 1e-4, torch.ones_like(std_feat), std_feat)

        z_feat = (x - mean_feat) / std_feat               # (batch, feature, time)

        # Apply learnable scale and bias
        l1 = self.l1.transpose(0, 1)                      # (1, time)
        B1 = self.B1.transpose(0, 1)                      # (1, time)
        X1 = l1.unsqueeze(1) * z_feat + B1.unsqueeze(1)    # (batch, feature, time)

        # Combine feature- and time-normalizations
        out = self.y1 * X1 + self.y2 * X2

        return out