import torch
from torch import nn
from torch.utils.data import Dataset


class SiameseNetwork(nn.Module):
    """3D ConvNet for 3DMatch

    Args:
        in_channel: input channel size
        patch: patch size
    """

    def __init__(self, in_channel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channel_size, 64, (3, 3, 3)),
            nn.Conv3d(64, 64, (3, 3, 3)),
            nn.MaxPool3d((2, 2, 2), 2),
            nn.Conv3d(64, 128, (3, 3, 3)),
            nn.Conv3d(128, 128, (3, 3, 3)),
            nn.Conv3d(128, 256, (3, 3, 3)),
            nn.Conv3d(256, 256, (3, 3, 3)),
            nn.Conv3d(512, 512, (3, 3, 3)),
            nn.Conv3d(512, 512, (3, 3, 3)),
        )

    def forward(self, patches):
        outputs = self.encoder(patches)
        return outputs


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()


class Preprocessing:
    def __init__(self) -> None:
        pass


class ThreeDMatchDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, idx):
        pass
