from typing import List
import torch
from torch import nn


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
        conv = nn.Conv1d(in_channels, out_channels, 1)
        nn.init.xavier_uniform_(conv.weight)
        module_list.append(conv)

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


class PointNetExtractor(nn.Module):
    """
    PointNet encoder: processing between inputs and Maxpooling.

    Args:
        use_input_transform: use of input transform net
        use_feature_transform: use of feature transform net
    """

    def __init__(
        self,
        use_input_transform: bool = True,
        use_feature_transform: bool = True,
    ):
        super().__init__()

        self.input_transform_net = InputTransformNet()
        self.encoder1 = nn.Sequential(
            MLP(3, 64),
            MLP(64, 64),
        )
        self.feature_transform_net = FeatureTransformNet(k=64)
        self.encoder2 = nn.Sequential(
            MLP(64, 64),
            MLP(64, 128),
            MLP(128, 1024),
        )

        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

    def forward(self, x: torch.Tensor):
        """Network processes

        Args:
            x: (B, C, N)

        Returns:
            x: global features, (B, 1024, 1)
            coord_trans: outputs of input transform network, (B, 3, 3)
            feature_trans: outputs of feature transform network, (B, 64, 64)
        """
        # transpose xyz
        if self.use_input_transform:
            coord_trans = self.input_transform_net(x)
            x = self.transpose(x, coord_trans)
        else:
            coord_trans = None

        x = self.encoder1(x)

        # transpose features
        if self.use_feature_transform:
            feat_trans = self.feature_transform_net(x)
            x = self.transpose(x, feat_trans)
        else:
            feat_trans = None

        x = self.encoder2(x)

        # get a global feature
        x = torch.amax(x, dim=2, keepdim=True)

        return (
            x,
            coord_trans,
            feat_trans,
        )

    def transpose(self, x, trans):
        x = torch.transpose(x, 1, 2)
        x = torch.bmm(x, trans)
        x = torch.transpose(x, 1, 2)
        return x


class PointNetClassification(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_input_transform: bool = True,
        use_feature_transform: bool = True,
    ):
        """
        PointNet for classification.
        Parameters
        ----------
        num_classes: int
            number of classes for predictions
        use_input_transform: bool
            use transform module for input point clouds
        use_feature_transform: bool
            use transform module for features
        """
        super().__init__()

        self.encoder = PointNetExtractor(
            use_input_transform=use_input_transform,
            use_feature_transform=use_feature_transform,
        )

        self.decoder = nn.Sequential(
            MLP(1024, 512, use_batch_norm=False),
            nn.Dropout(p=0.3),
            MLP(512, 256, use_batch_norm=False),
            nn.Dropout(p=0.3),
            MLP(256, num_classes),
        )

        self.num_classes = num_classes
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

    def forward(self, inputs):
        """
        PointNet predicts a class label of inputs.
        Parameters
        ----------
        inputs: torch.tensor
            point cloud (inputs.shape = (batch, channel, point))
        Returns
        -------
        pred_labels:torch.tensor
            prediction labels for point clouds (pred_labels.shape = (batch, class))
        """

        x, coord_trans, feat_trans = self.encoder(inputs)
        pred_labels = self.decoder(x)
        pred_labels = torch.squeeze(pred_labels, dim=2)

        return pred_labels, coord_trans, feat_trans


def sampling_layer(coords, num_samples):
    """
    Sampling layer in PointNet++
    Parameters
    ----------
    coords : torch.tensor [B, 3, N]
        xyz tensor
    num_samples : int
       number of samples for furthest point sample
    Return
    ------
    sampled_coords : torch.tensor [B, 3, num_samples]
        sampled xyz using furthest point sample
    """
    fps_idx = furthest_point_sampling(
        coords, num_samples
    )  # fps_idx = batch_fps(coords, num_samples)
    fps_idx = fps_idx.type(torch.long)
    sampled_coords = index2points(coords, fps_idx)
    return sampled_coords


def group_layer(coords, center_coords, num_samples, radius, points=None):
    """
    Group layer in PointNet++
    Parameters
    ----------
    coords : torch.tensor [B, 3, N]
        xyz tensor
    center_coords : torch.tensor [B, 3, N']
        xyz tensor of ball query centers
    num_samples : int
       maximum number of samples for ball query
    radius : float
        radius of ball query
    points : torch.tensor [B, C, N]
        Concatenate points to return value.
    Return
    ------
    new_points : torch.tensor [B, 3, N', num_samples] or [B, 3+C, N', num_samples]
        If points is not None, new_points shape is [B, 3+C, N', num_samples].
    """
    # Get sampled coords idx by ball query.
    idx = ball_query(center_coords, coords, radius, num_samples)
    idx = idx.type(torch.long)

    # Convert idx to coords
    grouped_coords = index2points(coords, idx)
    center_coords = torch.unsqueeze(center_coords, 3)
    grouped_coords_norm = grouped_coords - center_coords

    if points is not None:
        grouped_points = index2points(points, idx)
        new_points = torch.cat([grouped_coords_norm, grouped_points], dim=1)
        # note: PointNetSetAbstractionMsg is different order of concatenation.
        # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/2d08fa40635cc5eafd14d19d18e3dc646171910d/models/pointnet_util.py#L253
    else:
        new_points = grouped_coords_norm

    return new_points


# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L146
def sampling_and_group_layer_all(xyz, points=None):
    """
    Group layer (all)
    Parameters
    ----------
    xyz : torch.tensor [B, 3, N]
        xyz tensor
    points : torch.tensor [B, D, N]
        Concatenate points to return value (new_points).
    Returns
    -------
    new_xyz : torch.tensor [B, 3, 1]
        new xyz
    new_points : torch.tensor [B, 3, 1, N] or [B, 3+D, 1, N]
        If points is not None, new_points shape is [B, 3+D, 1, N].
    """
    device = xyz.device
    B, C, N = xyz.shape
    new_xyz = torch.zeros(B, C, 1).to(device)
    grouped_xyz = xyz.view(B, C, 1, N)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, -1, 1, N)], dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L166
class PointNetSetAbstraction(nn.Module):
    """
    PointNetSetAbstraction
    Parameters
    ----------
    num_fps_points : int
        number of samples for furthest point sample
    radius : float
        radius of ball query
    num_bq_points : int
        maximum number of samples for ball query
    init_in_channel : int
        input channel size
    mlp : list [O]
        MLP output channel sizes of PointNet layer
    group_all : bool
        group_all
    See Also
    --------
    O : out channel sizes
    """

    def __init__(
        self,
        num_sampling_points:int,
        radius:float,
        k:int,
        input_channel:int,
        mlp_channel_list: List[int],
        group_all:bool=False,
    ):
        # create MLP for a PointNet Layer
        layers = []
        current_channel = input_channel
        for out_channel in mlp_channel_list:
            layers.append(MLP(current_channel, out_channel))
            current_channel = out_channel
        self.mlp = nn.Sequential(*layers)

        self.num_sampling_points = num_sampling_points
        self.radius = radius
        self.k = k
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Parameters
        ----------
        xyz : torch.tensor [B, 3, N]
            xyz tensor
        points : torch.tensor [B, C, N]
            features of points
        Returns
        -------
        new_xyz : torch.tensor [B, 3, num_fps_points]
            sampled xyz using furthest point sample
        new_points : torch.tensor [B, D, num_fps_points]
            features of points by PointNetSetAbstraction
        """

        # Sampling Layer and Grouping Layer
        if self.group_all:
            new_xyz, new_points = sampling_and_group_layer_all(xyz, points)
        else:
            new_xyz = sampling_layer(xyz, self.num_fps_points)
            new_points = group_layer(
                xyz, new_xyz, self.num_bq_points, self.radius, points=points
            )

        # PointNet Layer
        new_points = self.mlp(new_points)
        new_points = torch.amax(new_points, -1)

        return new_xyz, new_points


class PointNet2SSGClassification(nn.Module):
    """
    PointNet++ with SSG for Classification
    Parameters
    ----------
    num_classes:int
        number of classes
    point_feature_size:int
        size of input feature other than xyz
    """

    def __init__(self, num_classes, point_feature_size=0):
        in_channel = 3 + point_feature_size
        self.sa1 = PointNetSetAbstraction(
            num_fps_points=512,
            radius=0.2,
            num_bq_points=32,
            init_in_channel=in_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            num_fps_points=128,
            radius=0.4,
            num_bq_points=64,
            init_in_channel=128 + 3,
            mlp=[128, 128, 128],
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
            Linear(1024, 512),
            nn.Dropout(0.5),
            Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes
        self.point_feature_size = point_feature_size

    def forward(self, inputs):
        B, C, N = inputs.shape

        if self.point_feature_size > 0:
            xyz = inputs[:, :3, :]
            features = inputs[:, 3:, :]
        else:
            xyz = inputs
            features = None

        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.decoder(x)

        return x
