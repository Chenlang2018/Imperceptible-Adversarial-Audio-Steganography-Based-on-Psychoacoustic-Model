import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Steganalyzer(nn.Module):
    def __init__(self):
        super(Steganalyzer, self).__init__()
        kernel = np.array([[1, -1, 0, 0, 0],
                           [1, -2, 1, 0, 0],
                           [1, -3, 3, -1, 0],
                           [1, -4, 6, -4, 1]], dtype=float)
        kernel = kernel.reshape((4, 1, 5)).astype(np.float32)
        kernel = torch.from_numpy(kernel)
        self.kernel = nn.Parameter(data= kernel, requires_grad=True)

        self.first_conv = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.group1 = nn.Sequential(
            nn.Conv1d(8, 8, 5, 1, 2),
            nn.Conv1d(8, 16, 1, 1, 0)  # batch,16,16384
        )
        self.group2 = nn.Sequential(
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,32,8192
        )
        self.group3 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,64,4096
        )
        self.group4 = nn.Sequential(
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,128,2048
        )
        self.group5 = nn.Sequential(
            nn.Conv1d(128, 128, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1, 1, 0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)  # batch,256,1024
        )
        self.group6 = nn.Sequential(
            nn.Conv1d(256, 256, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1, 1, 0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # batch,512,1
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.conv1d(x, self.kernel, padding=2)
        x = self.first_conv(x)
        x = torch.clamp(x, -3, 3)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = self.group6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

