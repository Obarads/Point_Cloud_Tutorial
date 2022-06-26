from typing import List
import torch
from torch import nn
from ..sampling import farthest_point_sampling
from ..operator import index2points
from ..nns import radius_and_k_nearest_neighbors


class MLP(nn.Module):
    """MLP module"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        # define a list for a MLP
        module_list = []

        # define a conv
        conv = nn.Conv2d(in_channels, out_channels, (1,1))
        nn.init.xavier_uniform_(conv.weight)
        module_list.append(conv)

        # define a batch norm
        if use_batch_norm:
            norm = nn.BatchNorm2d(out_channels)
            module_list.append(norm)

        # define an activation function
        if act is not None:
            module_list.append(act)

        # define MLP
        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)


class FCL(nn.Module):
    """FUlly connected layer module"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        # define a list for a FCL
        module_list = []

        # define a linear
        linear = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(linear.weight)
        module_list.append(linear)

        # define a batch norm
        if use_batch_norm:
            norm = nn.BatchNorm1d(out_channels)
            module_list.append(norm)

        # define an activation function
        if act is not None:
            module_list.append(act)

        # define MLP
        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)


def sampling_layer(coords: torch.tensor, num_samples: int) -> torch.Tensor:
    """
    Sampling layer in PointNet++

    Args:
        coords: xyz coordinates (B, 3, N)
        num_samples : number of samples for farthest point sample

    Return:
        sampled_coords: sampled xyz using farthest point sample (B, 3, num_samples)
    """
    fps_idx = farthest_point_sampling(coords, num_samples)
    fps_idx = fps_idx.type(torch.long)
    sampled_coords = index2points(coords, fps_idx)
    return sampled_coords


def group_layer(
    coords: torch.Tensor,
    center_coords: torch.Tensor,
    num_samples: int,
    radius: float,
    features: torch.Tensor = None,
) -> torch.Tensor:
    """
    Group layer in PointNet++

    Args:
        coords: xyz coordinats (B, 3, N)
        center_coords: query points (B, 3, N')
        num_samples: maximum number of nearest neighbors for ball query
        radius: radius of ball query
        features: point features, (B, C, N')

    Return:
        new_points: (B, 3, N', num_samples) or (B, 3+C, N', num_samples)
    """
    # Get sampled coords idx by ball query.
    idx, _ = radius_and_k_nearest_neighbors(
        center_coords, coords, num_samples, radius
    )
    # idx = idx.type(torch.long)

    # Convert idx to coords
    grouped_coords = index2points(coords, idx)
    center_coords = torch.unsqueeze(center_coords, 3)
    grouped_coords = grouped_coords - center_coords

    if features is not None:
        grouped_features = index2points(features, idx)
        new_features = torch.cat([grouped_coords, grouped_features], dim=1)
    else:
        new_features = grouped_coords

    return new_features


def grouping_all_points(coords: torch.Tensor, features: torch.Tensor = None):
    """Grouping all points for classification network

    Args:
        coords: xyz coordinates (B, 3, N)
        features: point features (B, D, N)
    Returns:
        new_points : torch.tensor [B, 3, 1, N] or [B, 3+D, 1, N]
    """
    B, C, N = coords.shape
    grouped_xyz = coords.view(B, C, 1, N)
    if features is not None:
        new_points = torch.cat([grouped_xyz, features.view(B, -1, 1, N)], dim=1)
    else:
        new_points = grouped_xyz
    return new_points


class PointNetSetAbstraction(nn.Module):
    """
    PointNetSetAbstraction

    Args:
        num_fps_points: number of samples for furthest point sample
        radius: radius of ball query
        num_bq_points: maximum number of samples for ball query
        init_in_channel: input channel size
        mlp: MLP output channel sizes of PointNet layer
        group_all: for classification
    """

    def __init__(
        self,
        num_fps_points: int,
        radius: float,
        num_bq_points: int,
        init_in_channel: int,
        mlp: List[int],
        group_all: bool,
    ):
        super().__init__()

        # create MLP for a PointNet Layer
        layers = []
        in_channel = init_in_channel
        for out_channel in mlp:
            layers.append(MLP(in_channel, out_channel))
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)

        self.num_fps_points = num_fps_points
        self.radius = radius
        self.num_bq_points = num_bq_points
        self.group_all = group_all

    def forward(self, coords, features):
        """
        Args:
            coords: (B, 3, N)
            features: (B, C, N) or None
        
        Returens:
            new_coords: sampled points
            new_features: new features
        """
        # Sampling Layer and Grouping Layer
        if self.group_all:
            new_coords = None
            new_features = grouping_all_points(coords, features)
        else:
            new_coords = sampling_layer(coords, self.num_fps_points)
            new_features = group_layer(
                coords, new_coords, self.num_bq_points, self.radius, features
            )

        # PointNet Layer
        new_features = self.mlp(new_features)
        new_features = torch.max(new_features, -1)[0]

        return new_coords, new_features


class PointNet2SSGClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            num_fps_points=512,
            radius=0.2,
            num_bq_points=32,
            init_in_channel=3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            num_fps_points=128,
            radius=0.4,
            num_bq_points=64,
            init_in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            num_fps_points=None,
            radius=None,
            num_bq_points=None,
            init_in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.decoder = nn.Sequential(
            FCL(1024, 512),
            nn.Dropout(0.5),
            FCL(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, coords, features):
        coords, features = self.sa1(coords, features)
        coords, features = self.sa2(coords, features)
        _, features = self.sa3(coords, features)

        B = len(features)
        x = features.view(B, 1024)
        x = self.decoder(x)

        return x
