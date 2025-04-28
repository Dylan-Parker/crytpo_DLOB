# Reference: /notebooks/CNN Deeplearning v2 (4).ipynb
# data processing and training available there
"""
This is my implementation of the architecture that is pictured in the appendix of the paper "Deep Learning for Digital Asset Limit Order Books"
by Jha et Al. This is the architecture that the main results in that paper are attributed to.
"""

import numpy as np
import torch.nn as nn
class CNNClassifier(nn.Module):
    """
    Conv1D → Residual Blocks → Pooling → AvgPool → Flatten → Dense → Output
    Input: (batch_size, sequence_length, num_features)
    """
    def __init__(self, config, input_shape, device, input_size):
        super().__init__()
        self.device = device
        self.batch_size = config.train.batch_size
        self.seq_length = input_shape[1]
        self.num_features = input_shape[2]

        # dilation and padding calculation
        self.dilation = int(np.log2((self.seq_length - 1) / (2 * (4 - 1)) + 1))
        total_pad = self.dilation * (4 - 1)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left

        # dropout probabilities
        conv_dp = getattr(config.model, 'conv_dropout', 0.2)
        fc_dp   = getattr(config.model, 'fc_dropout',   0.5)

        # conv + pool
        self.convblock0 = nn.Sequential(
            nn.ConstantPad1d((pad_left, pad_right), 0),
            nn.Conv1d(self.num_features, 16, kernel_size=4, bias=False, dilation=self.dilation),
            nn.MaxPool1d(2),
            nn.Dropout1d(conv_dp)
        )

        # residual conv blocks with dropout
        def make_block(ch):
            return nn.Sequential(
                nn.ConstantPad1d((pad_left, pad_right), 0),
                nn.Conv1d(ch, ch, kernel_size=4, bias=False, dilation=self.dilation),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Dropout1d(conv_dp),
                nn.ConstantPad1d((pad_left, pad_right), 0),
                nn.Conv1d(ch, ch, kernel_size=4, bias=False, dilation=self.dilation),
                nn.BatchNorm1d(ch),
                nn.Dropout1d(conv_dp)
            )

        self.convblock1 = make_block(16)
        self.convblock2 = make_block(16)
        self.convblock3 = make_block(16)

        # downsample block
        self.convblock4 = nn.Sequential(
            nn.ConstantPad1d((pad_left, pad_right), 0),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(conv_dp),
            nn.ConstantPad1d((pad_left, pad_right), 0),
            nn.Conv1d(32, 32, kernel_size=4, bias=False, dilation=self.dilation),
            nn.BatchNorm1d(32),
            nn.Dropout1d(conv_dp)
        )
        self.skipblock4 = nn.Sequential(
            nn.ConstantPad1d((pad_left, pad_right), 0),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(32)
        )

        self.convblock5 = make_block(32)

        # pooling and classifier
        self.pool2 = nn.AvgPool1d(kernel_size=(self.seq_length // 4))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Dropout(fc_dp),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # x: (B, T, F) → conv1d expects (B, F, T)
        x = x.permute(0, 2, 1)

        x = self.convblock0(x)
        res = x
        x = self.convblock1(x) + res
        x = nn.functional.relu(x)

        res = x
        x = self.convblock2(x) + res
        x = nn.functional.relu(x)

        res = x
        x = self.convblock3(x) + res
        x = nn.functional.relu(x)

        res = x
        x = self.convblock4(x) + self.skipblock4(res)
        x = nn.functional.relu(x)

        res = x
        x = self.convblock5(x) + res
        x = nn.functional.relu(x)

        x = self.pool2(x)
        x = self.flatten(x)
        return self.fc(x)

    def set_train_model(self):
        self.mode = "train"

    def set_eval_mode(self):
        self.mode = "eval"

    def set_test_mode(self):
        self.mode = "test"
