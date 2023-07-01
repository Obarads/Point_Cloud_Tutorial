from typing import List
import torch
from torch import nn
from ..sampling import farthest_point_sampling
from ..operator import index2points
from ..nns import radius_and_k_nearest_neighbors, k_nearest_neighbors


class MLP(nn.Module):
    """MLP module"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        act: nn.Module = nn.ReLU(inplace=True),
        conv_dim: int = 2,
    ):
        super().__init__()

        # define a list for a MLP
        module_list = []

        # define a conv
        if conv_dim == 1:
            conv = nn.Conv1d(in_channels, out_channels, 1)
        elif conv_dim == 2:
            conv = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            raise ValueError("conv_dim must be 1 or 2")

        nn.init.xavier_uniform_(conv.weight)
        module_list.append(conv)

        # define a batch norm
        if use_batch_norm:
            if conv_dim == 1:
                norm = nn.BatchNorm1d(out_channels)
            elif conv_dim == 2:
                norm = nn.BatchNorm2d(out_channels)
            else:
                raise ValueError("conv_dim must be 1 or 2")
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
    grouped_coords = coords.view(B, C, 1, N)
    if features is not None:
        new_points = torch.cat(
            [grouped_coords, features.view(B, -1, 1, N)], dim=1
        )
    else:
        new_points = grouped_coords
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


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, init_in_channel, mlp):
        super().__init__()

        layers = []
        in_channel = init_in_channel
        for out_channel in mlp:
            layers.append(MLP(in_channel, out_channel, conv_dim=1))
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)
        self.interpolation_k = 3

    def forward(self, coords_1, coords_2, features_1, features_2):
        _, _, N = coords_1.shape
        _, _, S = coords_2.shape

        if S == 1:
            interpolated_features = features_1.repeat(1, 1, N)
        else:
            idxs, dists = k_nearest_neighbors(
                coords_1, coords_2, self.interpolation_k
            )
            interpolated_features = self.distance_weight_interpolation(
                features_2, idxs, dists
            )

        if features_1 is not None:
            new_features = torch.cat([features_1, interpolated_features], dim=1)
        else:
            new_features = interpolated_features

        new_features = self.mlp(new_features)

        return new_features

    def distance_weight_interpolation(self, features, indices, distances):
        B, N, _ = indices.shape
        distance_reciprocal = 1.0 / torch.maximum(
            distances, torch.full_like(distances, fill_value=1e-10)
        )
        norm = torch.sum(distance_reciprocal, dim=2, keepdim=True)
        weight = distance_reciprocal / norm
        weigted_features = index2points(features, indices) * weight.view(
            B, 1, N, self.interpolation_k
        )
        interpolated_features = torch.sum(weigted_features, dim=3)
        return interpolated_features


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


class PointNet2SSGSemanticSegmentation(nn.Module):
    """
    PointNet++ with SSG for Semantic segmentation
    Parameters
    ----------
    num_classes:int
        number of classes
    point_feature_size:int
        feature size other than xyz
    """

    def __init__(self, num_classes, point_feature_size=0):
        super().__init__()
        in_channel = 3 + point_feature_size
        self.sa1 = PointNetSetAbstraction(
            num_fps_points=1024,
            radius=0.1,
            num_bq_points=32,
            init_in_channel=in_channel,
            mlp=[32, 32, 64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            num_fps_points=256,
            radius=0.2,
            num_bq_points=32,
            init_in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            num_fps_points=64,
            radius=0.4,
            num_bq_points=32,
            init_in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            num_fps_points=16,
            radius=0.8,
            num_bq_points=32,
            init_in_channel=256 + 3,
            mlp=[256, 256, 512],
            group_all=False,
        )
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channel, [128, 128, 128])

        self.output_layers = nn.Sequential(
            MLP(128, 128),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

        self.point_feature_size = point_feature_size

    def forward(self, coords: torch.tensor, features: torch.tensor = None):
        l1_coords, l1_features = self.sa1(coords, features)
        l2_coords, l2_features = self.sa2(l1_coords, l1_features)
        l3_coords, l3_features = self.sa3(l2_coords, l2_features)
        l4_coords, l4_features = self.sa4(l3_coords, l3_features)

        l3_features = self.fp4(l3_coords, l4_coords, l3_features, l4_features)
        l2_features = self.fp3(l2_coords, l3_coords, l2_features, l3_features)
        l1_features = self.fp2(l1_coords, l2_coords, l1_features, l2_features)
        l0_features = self.fp1(coords, l1_coords, features, l1_features)

        x = self.output_layers(l0_features)

        return x
