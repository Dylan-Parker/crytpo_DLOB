# Reference: /notebooks/CNN Deeplearning v2 (4).ipynb
# data processing and training available there

from models.base_model import BaseModel
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.utils import print_model_info

class CNNClassifier(nn.Module):
    """
    Conv2D → Reshape → Conv1D → MaxPool1D → Conv1D → MaxPool1D → BiLSTM → Dense → Dense → Output
    Input: (batch_size, 100, 40, 1)
    """
    def __init__(self, config, input_shape, device, input_size):
        super().__init__()
        # Define CNN layers based on crypto_lob.pdf / basic_cnn_model.ipynb [cite: 1, 5, 39, 94]
        # e.g., Conv2D -> Reshape -> Conv1D -> Pooling -> Dense [cite: 1, 5]
        self.device = device
        self.batch_size = config.train.batch_size
        #self.hidden_size=config.model.hidden_size
        self.seq_length = input_shape[1]
        self.num_features = input_shape[2]
        # First Conv2D layer, input (batch_size, 1, 100, 40) -> output (batch_size, 16, 97, 1)
        self.conv1d_0 = nn.Conv1d(in_channels=self.num_features, out_channels=16, kernel_size=4, padding='same', bias=False)
        self.pool1 = nn.MaxPool1d(2)

        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(16),
        )

        self.a1 = nn.ReLU()

        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(16),
        )

        self.a2 = nn.ReLU()

        self.convblock3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(16),
        )

        self.a3 = nn.ReLU()
        self.convblock4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(32),
        )
        self.skipblock4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(32),
        )
        self.a4 = nn.ReLU()
        self.convblock5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, padding='same', bias=False),
            nn.BatchNorm1d(32),
        )
        self.a5 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=(self.seq_length//2//2)-1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(32, 3)  # 3 output classes for softmax

        self.name = "CNN Model"
        self.input_size = input_size
        # print out the model architecture when first intializing
        print(f'Model: {self.name}')
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {self.name}")
        print(f"\n🧠 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        ## todo: get the summary working properly with the right input shape
        # print_model_info(self, input_size, self.name)

    def forward(self, x):
        # Conv2D
        x.to(self.device)
        #print(x.shape)
        x = self.conv1d_0(x)
        x = self.pool1(x)
        #print(x.shape)
        x = self.convblock1(x) + x
        x = self.a1(x)
        #print(x.shape)
        x = self.convblock2(x) + x
        x = self.a2(x)
        #print(x.shape)
        x = self.convblock3(x) + x
        x = self.a3(x)
        #print(x.shape)
        x = self.convblock4(x) + self.skipblock4(x)
        x = self.a4(x)
        #print(x.shape)
        x = self.convblock5(x) + x
        x = self.a5(x)
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc(x)
        return x

    def set_train_model(self):
        self.mode = "train"

    def set_eval_mode(self):
        self.mode = "eval"

    def set_test_mode(self):
        self.mode = "test"
