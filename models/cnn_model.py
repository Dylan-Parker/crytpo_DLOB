# Reference: /notebooks/CNN Deeplearning v2 (4).ipynb
# data processing and training available there

from models.base_model import BaseModel
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    Conv2D → Reshape → Conv1D → MaxPool1D → Conv1D → MaxPool1D → BiLSTM → Dense → Dense → Output
    Input: (batch_size, 100, 40, 1)
    """
    def __init__(self, config, input_shape, device):
        super().__init__()
        # Define CNN layers based on crypto_lob.pdf / basic_cnn_model.ipynb [cite: 1, 5, 39, 94]
        # e.g., Conv2D -> Reshape -> Conv1D -> Pooling -> Dense [cite: 1, 5]
        self.device = device
        self.batch_size = config.train.batch_size
        #self.hidden_size=config.model.hidden_size
        self.seq_length = input_shape[1]
        self.num_features = input_shape[2]
        # First Conv2D layer, input (batch_size, 1, 100, 40) -> output (batch_size, 16, 97, 1)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, self.num_features))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # Reshape layer (to reshape for Conv1D)
        #self.T = config("T")  # Store T for dynamic reshaping
        # First Conv1D layer
        self.conv1d_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4)
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(16)
        # MaxPooling1D
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        # Second Conv1D layer
        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        # Batch Normalization
        self.bn2 = nn.BatchNorm1d(32)
        # MaxPooling1D
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, bidirectional=True, batch_first=True)
        # Dense layers
        self.fc1 = nn.Linear(64 * 2, 32)  # *2 because it's bidirectional
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 output classes for softmax

    def forward(self, x):
        # Conv2D
        x.to(self.device)
        x = x.unsqueeze(1)
        x = self.conv2d(x)
        x = self.leaky_relu(x)
        # Reshape from Conv2D output to match Conv1D input
        x = x.view(x.shape[0], 16, -1)
        # First Conv1D
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        # Batch Normalization + MaxPooling
        #x = self.bn1(x)
        x = self.maxpool1(x)
        # Second Conv1D
        x = self.conv1d_2(x)
        x = self.leaky_relu(x)
        # Batch Normalization + MaxPooling
        x = self.bn2(x)
        x = self.maxpool2(x)
        # Check the shape before LSTM
        #print("Shape before LSTM:", x.shape) # Print the shape before entering LSTM
        # Bidirectional LSTM
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last time step
        # Dense layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        # Softmax output
        # x = F.softmax(x, dim=-1)
        return x
