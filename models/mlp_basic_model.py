from torch import nn

from models.base_model import BaseModel
from utils.utils import print_model_info

class BasicMLPModel(BaseModel):
    def __init__(self, config, device, input_size):
        super().__init__(config, device, input_size)
        self.mode = "train"
        self.device = device
        self.input_size = input_size
        self.name = "Basic MLP Model"

        self.in_features = config.model.in_features
        #self.out_features = config.model.out_features
        self.layer1 = nn.Linear(in_features=self.in_features, out_features=3)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

        # print out the model architecture when first intializing
        print_model_info(self, input_size, self.name)

    def set_train_model(self):
        self.mode = "train"

    def set_eval_mode(self):
        self.mode = "eval"

    def set_test_mode(self):
        self.mode = "test"

    def forward(self, x):
        # print("Calling MLP Forward")  # use for debugging
        #Preserve Batch dim when passing to FC Layer
        out = self.layer1(x.view(x.size(0), -1))
        out = self.relu(out)
        #out = self.sigmoid(out)
        return out
