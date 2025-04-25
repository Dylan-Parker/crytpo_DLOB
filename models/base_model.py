# models/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn # Or tensorflow.keras.Model
from torchinfo import summary
from utils.utils import print_model_info
import torch
class BaseModel(nn.Module, ABC):
    def __init__(self, config, device, input_size):
        super().__init__()
        self.config = config
        self.train_mode = True
        self.device = device
        self.input_size = input_size
        self.name = "Base Model"

        self.in_features = config.model.in_features

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=3)
        self.relu = nn.ReLU()
        # print out the model architecture when first intializing
        # print_model_info(self, input_size, self.name)

    # @abstractmethod
    def forward(self, x): # -> model_output (logits/predictions)
        print("Calling Base Forward")
        assert x.device == next(self.parameters()).device, "CPU↔GPU mismatch"
        assert x.dtype in (torch.float16, torch.float32), "Bad dtype for MPS"
        assert x.is_contiguous(), "Make contiguous()"
        # 1) move to the model’s device
        x = x.to(self.fc1.weight.device)  # cpu→mps or cpu→cuda
        # 2) cast to fp32 (MPS only handles fp16 / fp32)
        x = x.to(torch.float32)
        # 3) make memory contiguous before view/Linear
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc1(x.view(x.size(0), -1))
        # out = self.relu(out)
        return out

    def set_train_model(self):
        self.mode = "train"

    def set_eval_mode(self):
        self.mode = "eval"

    def set_test_mode(self):
        self.mode = "test"
