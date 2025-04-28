"""
CREDIT FOR ORIGINAL CODE FROM Leonardo Berti; TLOB Research Paper Repo
https://github.com/LeonardoBerti00/TLOB/blob/main/models/deeplob.py
Original Architecture from Zhang et Al "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
https://arxiv.org/abs/1808.03668
Added Dropout regularization after BatchNorm layers and before final FC.
"""

from torch import nn
import torch


class DeepLOB(nn.Module):
    def __init__(self, config, device, input_size):
        super().__init__()
        self.config = config
        self.device = device
        self.input_size = input_size
        self.mode = None
        self.name = "DeepLOB Model"

        # dropout rates from config or defaults
        conv_dp = self.config.model.conv_dropout or 0.2
        fc_dp   = self.config.model.fc_dropout or 0.5
        negative_slope = self.config.model.neg_slope or 0.01
        # convolution blocks with Dropout2d
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,2)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(32),
            nn.Dropout2d(conv_dp),
        )

        # inception modules with Dropout2d
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(64),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(64),
            nn.Dropout2d(conv_dp),
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(64),
            nn.Dropout2d(conv_dp),
            nn.Conv2d(64, 64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(64),
            nn.Dropout2d(conv_dp),
        )

        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(32, 64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope),
            nn.BatchNorm2d(64),
            nn.Dropout2d(conv_dp),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64,
                            num_layers=1, batch_first=True)
        # final fc with Dropout
        self.fc1 = nn.Sequential(
            nn.Dropout(fc_dp),
            nn.Linear(64, 3)
        )

        # print architecture
        print(f'Model: {self.name}')
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🧠 Total parameters: {total:,}")
        print(f"🎯 Trainable parameters: {trainable:,}")

    def forward(self, x):
        # ensure shape (B, T, F)
        if x.dim()==3 and x.shape[1]==self.input_size[1]:
            x = x.permute(0,2,1)
        x = x[:, None, :, :]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x1 = self.inp1(x)
        x2 = self.inp2(x)
        x3 = self.inp3(x)
        x = torch.cat((x1, x2, x3), dim=1)

        x = x.permute(0,2,1,3)
        x = x.reshape(-1, x.shape[1], x.shape[2])

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc1(out)

    def set_train_model(self):
        self.mode="train"

    def set_eval_mode(self):
        self.mode="eval"

    def set_test_mode(self):
        self.mode="test"
