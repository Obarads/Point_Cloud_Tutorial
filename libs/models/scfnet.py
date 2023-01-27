import torch
from torch import nn
from libs.modules.scf_module import SCFModule
from libs.modules.layer import PointwiseConv1D

class SCFNet(nn.Model):
    def __init__(self, , layer_config) -> None:
        super().__init__()

        layer_config = [
            4, 1, 2, 3
        ]

        f_encoder_list = []
        for i, lc in range(1, len(layer_config)):
            f_encoder_list.append(SCFModule(lc-1, lc))

        mlp_1 = PointwiseConv1D()

        f_decoder_list = []
        for i, lc


