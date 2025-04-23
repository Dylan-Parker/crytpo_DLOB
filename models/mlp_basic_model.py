from torch import nn

from models.base_model import BaseModel

class BasicMLPModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config)
        self.train_mode = True

        self.in_features = config.model.in_features
        self.out_features = config.model.out_features
        self.layer1 = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten()
        out = self.layer1(x)
        out = self.relu(out)
        out = self.sigmoid(out)
        return out
