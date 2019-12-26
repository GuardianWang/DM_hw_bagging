import torch
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, num_class: int=17):
        super(BaseNet, self).__init__()
        self.num_class = num_class

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x):
        # x: 64 * 64
        x = self.layer1(x)  # (N, 32, 32, 32)
        x = self.layer2(x)  # (N, 64, 16, 16)
        x = self.layer3(x)  # (N, 128, 8, 8)
        x = self.layer4(x)  # (N, 256, 4, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
