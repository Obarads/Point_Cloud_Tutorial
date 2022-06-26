import torch
from torch import nn
from urllib3 import Retry

class VoxNetClassification(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 5, 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv3d(32, 32, 3, 1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear((6**3)*32, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.max_pool3d(x, 2, 2)
        x = x.reshape(-1, (6**3)*32)
        y = self.decoder(x)

        return y
