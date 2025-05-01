# models/base_model.py
from models.base_model import BaseModel
import torch
import torch.nn as nn

# RNNModel and LSTMModel
class RNNModel(BaseModel):
    def __init__(self, config, device, input_size):
        super().__init__(config, device, input_size)
        self.in_features = config.model.in_features
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.rnn = nn.RNN(
            input_size=self.in_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.hidden_dim, 3)

    def forward(self, x):
        x = x.to(self.device).to(torch.float32).contiguous()
        out, _ = self.rnn(x)
        out = self.fc_out(out[:, -1, :])
        return out


class LSTMModel(BaseModel):
    def __init__(self, config, device, input_size):
        super().__init__(config, device, input_size)
        self.in_features = config.model.in_features
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.hidden_dim, 3)

    def forward(self, x):
        x = x.to(self.device).to(torch.float32).contiguous()
        out, _ = self.lstm(x)
        out = self.fc_out(out[:, -1, :])
        return out
