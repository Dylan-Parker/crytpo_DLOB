# Reference: notebooks/copy of DepthANalysis (1).ipynb
# data  processing, model training and evaluation available there 

from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from pytorch_tcn import TCN

class TCNClassifier(nn.Module):
    def __init__(self, config):
    # def __init__(self, depth, output_size=3, dropout=0.4):
        super(TCNClassifier).__init__(self, config)
        self.depth = config("depth")
        self.output_size = config("output_size")
        self.dropout = config("dropout")
        # Define the TCN layer
        self.tcn = TCN(
            num_inputs=depth,
            num_channels=[64,64,64,64,64,64,64],
            kernel_size=7,
            dilations=[1, 2, 4, 8, 16, 32, 64], # Adjust dilations if needed
            dropout=self.dropout, # Provide dropout rate here
            causal=True,  # Padding type (causal is common for time series)
            use_norm='weight_norm',
            activation= 'relu',
            kernel_initializer='xavier_uniform',
            use_skip_connections=True, # Enable skip connections
            input_shape='NCL',
            embedding_mode='add',
            use_gate=False,
            lookahead=0
        )

        # Final linear layer (Dense in TensorFlow)
        self.fc = nn.Linear(64, self.output_size)

    def forward(self, x):
        #print("x.shape at beginning of forward:", x.shape)
        x = x.permute(0, 2, 1)  # Adjust shape to match PyTorch TCN
        x = self.tcn(x)
        #print("X.shape after tcn:", x.shape)        
        x = x[:, :, -1]  # Take the last output for classification
        #print("X.shape before fc:", x.shape)
        x = self.fc(x)  # Final classification layer
        return x  # Softmax is applied during loss computation

# Function to generate model
def generate_model(self.depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = TCNClassifier(depth=depth).to(device)
    # Define optimizer (Adam with learning rate 0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Loss function (equivalent to sparse_categorical_crossentropy)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion
