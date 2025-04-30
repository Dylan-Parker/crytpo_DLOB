# Reference: notebooks/copy of DepthANalysis (1).ipynb
# data processing, model training and evaluation available there 

#from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from pytorch_tcn import TCN

class TCNClassifier(nn.Module):
    def __init__(self, config, input_shape, device, input_size):
    # def __init__(self, depth, output_size=3, dropout=0.4):
        super().__init__()
        self.output_size = 3
        self.dropout = 0.4
        self.device = device
        print("device used in trainer:", device)
        self.batch_size = config.train.batch_size
        print("input.shape:", input_shape)
        self.seq_length = input_shape[1]
        self.depth = input_shape[2]
        self.name = "TCN model"
        
        # Define the TCN layer
        self.tcn = TCN(
            num_inputs=self.depth,
            num_channels=[16, 16, 16, 16], #[64,64,64,64,64,64,64],
            kernel_size=3, #7,
            dilations=[1, 2, 4, 8], # 16, 32, 64], # Adjust dilations if needed
            dropout=self.dropout, # Provide dropout rate here
            causal=True,  # Padding type (causal is common for time series)
            use_norm='weight_norm',
            activation= 'relu',
            kernel_initializer='xavier_uniform',
            use_skip_connections=True, # Enable skip connections
            input_shape='NCL', #(batch_size, num_channels, sequence_length)
            embedding_mode='add',
            use_gate=False,
            lookahead=0
        )

        # Final linear layer
        self.fc = nn.Linear(in_features=64, out_features=self.output_size)

        # print out the model architecture when first intializing
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {self.name}")
        print(f"\n🧠 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        ## todo: get the summary working properly with the right input shape
        # print_model_info(self, input_size, self.name)

    def forward(self, x):
        #print("x.shape at beginning of forward:", x.shape)
        #x = x.permute(0, 2, 1)  # Adjust shape to match PyTorch TCN
        x = self.tcn(x)
        #print("X.shape after tcn:", x.shape)        
        x = x[:, :, -1]  # Take the last output for classification
        #print("X.shape before fc:", x.shape)
        x = self.fc(x)  # Final classification layer
        return x  # Softmax is applied during loss computation

    def set_train_model(self):
        self.mode = "train"

    def set_eval_mode(self):
        self.mode = "eval"

    def set_test_mode(self):
        self.mode = "test"
        
"""
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
"""