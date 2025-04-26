import torch
import torch.nn as nn
from models.base_model import BaseModel
from utils.utils import print_model_info

class MLP(nn.Module):
    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x


class MLPLOB(BaseModel):
    def __init__(self, config, device, input_size):
        super().__init__(config, device, input_size)

        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.sequence_length = config.sequence_length

        self.proj = nn.Linear(self.input_size[1], self.hidden_dim)

        self.feature_mixing_mlps = nn.ModuleList()
        self.temporal_mixing_mlps = nn.ModuleList()

        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                self.feature_mixing_mlps.append(
                    MLP(
                        start_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim * 4,
                        final_dim=self.hidden_dim,
                    )
                )
                self.temporal_mixing_mlps.append(
                    MLP(
                        start_dim=self.sequence_length,
                        hidden_dim=self.sequence_length * 4,
                        final_dim=self.sequence_length,
                    )
                )
            else:
                # LAST layer shrinks dims
                self.feature_mixing_mlps.append(
                    MLP(
                        start_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim * 2,
                        final_dim=self.hidden_dim // 4,
                    )
                )
                self.temporal_mixing_mlps.append(
                    MLP(
                        start_dim=self.sequence_length,
                        hidden_dim=self.sequence_length * 2,
                        final_dim=self.sequence_length // 4,
                    )
                )

        self.hidden_dim = self.hidden_dim // 4
        self.sequence_length = self.sequence_length // 4

        # Now set up final_layers
        total_dim = self.hidden_dim * self.sequence_length
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4

        self.head = nn.Linear(total_dim, 3)

        self.name = "MLPLOB (adapted full version)"
        
        
    def forward(self, x):
        assert x.device == next(self.parameters()).device, "CPU↔GPU mismatch"
        assert x.dtype in (torch.float16, torch.float32), "Bad dtype for MPS"
        x = x.to(torch.float32).contiguous()

        # Input shape: (batch_size, feature_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        x = self.proj(x)  # project feature_dim to hidden_dim

        for feature_mlp, temporal_mlp in zip(self.feature_mixing_mlps, self.temporal_mixing_mlps):
            x = feature_mlp(x)  # Feature Mixing
            x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
            x = temporal_mlp(x)  # Temporal Mixing
            x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)

        # After last mixing, we now have (batch_size, seq_len//4, hidden_dim//4)

        # Flatten (batch_size, seq_len * hidden_dim)
        x = x.reshape(x.size(0), -1)

        # Pass through final_layers
        for layer in self.final_layers:
            x = layer(x)

        # Final classification head
        logits = self.head(x)

        return logits
