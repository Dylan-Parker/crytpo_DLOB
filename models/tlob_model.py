import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from models.base_model import BaseModel
from models.bin import BiN
from models.mlplob_model import MLP

# reference: https://github.com/LeonardoBerti00/TLOB/blob/main/models/tlob.py
# ComputeQKV module: Projects input to query, key, and value tensors
class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)

# TransformerLayer: One block of attention + MLP + skip connection + layer norm
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads, batch_first=True)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)  # or 0.2 #add dropout to prevent overfitting

    def forward(self, x):
        res = x
        q, k, v = self.qkv(x)
        attn_out, attn_weights = self.attention(q, k, v, need_weights=True, average_attn_weights=False)
        x = self.proj(attn_out)
        x = self.dropout(x)   # add dropout to prevent overfitting
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, attn_weights

# Sinusoidal positional encoding function
def sinusoidal_positional_embedding(seq_len, dim, n=10000.0):
    positions = torch.arange(0, seq_len).unsqueeze(1)
    denominators = torch.pow(n, 2 * torch.arange(0, dim // 2) / dim)
    embeddings = torch.zeros(seq_len, dim)
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)
    return embeddings

# Main TLOB Model
class TLOB(BaseModel):
    def __init__(self, config, device, input_size):
        super().__init__(config, device, input_size)

        # Load model hyperparameters from config
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.seq_size = config.sequence_length
        self.num_features = config.model.in_features
        self.num_heads = config.model.num_heads
        self.is_sin_emb = config.model.is_sin_emb
        self.dropout = config.model.dropout

        
        self.input_preprocessor = nn.Sequential(
            BiN(self.num_features, self.seq_size),
            Rearrange('b f s -> b s f'),
            nn.Linear(self.num_features, self.hidden_dim)
        )

        # Positional encoding (sinusoidal or learned)
        if self.is_sin_emb:
            pe = sinusoidal_positional_embedding(self.seq_size, self.hidden_dim)
            self.register_buffer('pos_encoder', pe)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_size, self.hidden_dim))

        # Build Transformer layers alternating attention across sequence and features
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                self.layers.append(TransformerLayer(self.hidden_dim, self.num_heads, self.hidden_dim, self.dropout))
                self.layers.append(TransformerLayer(self.seq_size, self.num_heads, self.seq_size, self.dropout))
            else:
                self.layers.append(TransformerLayer(self.hidden_dim, self.num_heads, self.hidden_dim // 4, self.dropout))
                self.layers.append(TransformerLayer(self.seq_size, self.num_heads, self.seq_size // 4, self.dropout))

        # Final MLP layers to project to output
        total_dim = (self.hidden_dim // 4) * (self.seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))  # 3 output classes

        self.name = "TLOB"

    def forward(self, x, store_att=False):
        # Device, dtype, and memory checks
        assert x.device == next(self.parameters()).device, "CPU↔GPU mismatch"
        assert x.dtype in (torch.float16, torch.float32), "Bad dtype for MPS"
        assert x.is_contiguous(), "Make contiguous()"

        # x is of shape (batch_size, num_features, seq_size)
        # Preprocess input: normalization, rearrange, and projection
        x = self.input_preprocessor(x)
        x = x + self.pos_encoder

        attn_maps = [] if store_att else None

        # Pass through Transformer layers, permute only after every layer
        for idx, layer in enumerate(self.layers):
            x, attn_weights = layer(x)
            if store_att:
                attn_maps.append(attn_weights.detach().cpu())  # Save attention maps
            x = x.permute(0, 2, 1)

        # Flatten and project to final output space
        x = rearrange(x, 'b s f -> b (f s)')
        for layer in self.final_layers:
            x = layer(x)

        if store_att:
            return x, attn_maps
        else:
            return x
