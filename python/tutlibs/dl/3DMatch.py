import torch
from torch import nn

class SiameseNetwork(nn.Module):
    """3D ConvNet for 3DMatch
    
    Args:
        in_channel: input channel size
        patch: patch size
    """
    def __init__(self, in_channel_size:int=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channel_size, 64, (3,3,3)),
            nn.Conv3d(64, 64, (3,3,3)),
            nn.MaxPool3d((2,2,2), 2),
            nn.Conv3d(64, 128, (3,3,3)),
            nn.Conv3d(128, 128, (3,3,3)),
            nn.Conv3d(128, 256, (3,3,3)),
            nn.Conv3d(256, 256, (3,3,3)),
            nn.Conv3d(512, 512, (3,3,3)),
            nn.Conv3d(512, 512, (3,3,3))
        )

    def forward(self, patches):
        outputs = self.encoder(patches)
        return outputs

class ThreeDMatch:
    def __init__(self, patch_size:int=30, in_channel_size:int=3) -> None:
        self.net = SiameseNetwork(in_channel_size)

        self.patch_size = patch_size

    def __call__(self, point_cloud: torch.Tensor):
        return

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()


