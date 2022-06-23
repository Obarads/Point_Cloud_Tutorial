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
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            1,
        )
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


class InputTransformNet(nn.Module):
    """
    Transform network for XYZ coordinate.
    """

    def __init__(self):
        super().__init__()

        # layers before a max-pooling
        self.encoder = nn.Sequential(
            MLP(3, 64),
            MLP(64, 128),
            MLP(128, 1024),
        )

        # layers after a max-pooling
        self.decoder = nn.Sequential(
            FCL(1024, 512),
            FCL(512, 256),
            nn.Linear(256, 9),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.amax(x, dim=2)
        x = self.decoder(x)
        x = torch.reshape(x, (-1, 3, 3))
        x += torch.eye(3, dtype=x.dtype, device=x.device)[None, :, :]
        return x


class FeatureTransformNet(nn.Module):
    """
    Transform network for features.
    """

    def __init__(self, k=64):
        super().__init__()

        # layers before a max-pooling
        self.encoder = nn.Sequential(
            MLP(k, 64),
            MLP(64, 128),
            MLP(128, 1024),
        )

        # layers after a max-pooling
        self.decoder = nn.Sequential(
            FCL(1024, 512),
            FCL(512, 256),
            nn.Linear(256, k * k),
        )

        # args
        self.k = k

    def forward(self, x):
        x = self.encoder(x)
        x = torch.amax(x, dim=2)
        x = self.decoder(x)
        x = torch.reshape(x, (-1, self.k, self.k))
        x += torch.eye(self.k, dtype=x.dtype, device=x.device)[None, :, :]
        return x


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
            x: global features, (B, 1024)
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
        x = torch.amax(x, dim=2)

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
            FCL(1024, 512),
            nn.Dropout(p=0.3),
            FCL(512, 256),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
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
        output = self.decoder(x)

        return output, coord_trans, feat_trans
