# models/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn # Or tensorflow.keras.Model

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x): # -> model_output (logits/predictions)
        # x: Input batch tensor (batch_size, time_steps, features)
        pass

    def train(self):
        # special training logic / considerations
        pass

    def eval(self):
        # special eval logic / considerations
        pass
