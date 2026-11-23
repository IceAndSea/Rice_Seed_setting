# ultralytics/models/sr/simplesr.py

import torch
import torch.nn as nn

class SimpleSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleSR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()  # 👈 加上这行
        )

    def forward(self, x):
        return self.net(x)

