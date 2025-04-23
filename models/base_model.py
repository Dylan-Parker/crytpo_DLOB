# models/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn # Or tensorflow.keras.Model

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_mode = True

    @abstractmethod
    def forward(self, x): # -> model_output (logits/predictions)
        # x: Input batch tensor (batch_size, time_steps, features)
        out = None

        return out

    def train(self):
        # special training logic / considerations
        self.train_mode = True
        return

    def eval(self):
        # special eval logic / considerations
        self.train_mode = False
        return
