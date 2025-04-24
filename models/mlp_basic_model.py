from torch import nn

from models.base_model import BaseModel

class BasicMLPModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config)
        self.mode = "train"
        self.device = device
        self.in_features = config.model.in_features
        #self.out_features = config.model.out_features
        self.layer1 = nn.Linear(in_features=self.in_features, out_features=3)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def forward(self, x):
        #Preserve Batch dim when passing to FC Layer
        out = self.layer1(x.view(x.size(0), -1))
        out = self.relu(out)
        #out = self.sigmoid(out)
        return out
