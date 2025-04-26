
from models.base_model import BaseModel
import torch
import torch.nn as nn

#linear logistic regression model
class LinearModel(BaseModel):
     def __init__(self, config, device, input_size):
        super().__init__(config, device, input_size)
        # Define linear layers as the numner of features x time steps so the model can see still sequence of features for fair comparison
        self.mode = "train"
        self.name = "Basic Linear Model"
        self.device = device
        self.input_size = input_size[1]
        self.sequence_length = config.sequence_length
        #batch size is not included in the input size
        #linear layer takes in the number of features x time steps
        self.linear = nn.Linear(in_features=self.input_size*self.sequence_length, out_features=3)
        # Cast weights and biases to float32
        self.linear.weight = nn.Parameter(self.linear.weight.type(torch.float32))
        self.linear.bias = nn.Parameter(self.linear.bias.type(torch.float32))
        
     def forward(self, x):
         # reshape x to (batch_size, num_features * sequence_length)
         x = x.view(x.size(0), -1)
         logits = self.linear(x.type(torch.float32)) # Cast input to float32
         probs = torch.softmax(logits, dim=1)
         return probs
     
     def train(self):
        self.mode = "train"

     def eval(self):
        self.mode = "eval"

     def test(self):
        self.mode = "test"
     

