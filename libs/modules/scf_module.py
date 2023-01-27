import torch
from torch import nn

from libs.modules.layer import PointwiseConv1D


class SCFModule(nn.Module):
    def __init__(self, in_channel_size:int, out_channel_size:int) -> None:
        super().__init__()

        f_pc_out_channel_size =out_channel_size // 2
        self.f_pc = PointwiseConv1D(in_channel_size, f_pc_out_channel_size)
        self.f_lc = LocalContextLearning(f_pc_out_channel_size, out_channel_size)
        f_gc_out_channel_size = out_channel_size * 2
        self.f_lc_conv = PointwiseConv1D(_, f_gc_out_channel_size, act=None, bn=False)
        self.shortcut = PointwiseConv1D(in_channel_size, f_gc_out_channel_size, act=None)

        self.f_gc = PointwiseConv1D(_, f_gc_out_channel_size, act=None)
        self.last_act = nn.LeakyReLU()

    def forward(self, x, xyz):

        x_f = self.f_pc(x)
        x_f, lg_volume_ratio = self.f_lc(x_f)
        x_f = self.f_lc_conv(x_f)
        sc = self.shortcut(x)
        x_g = torch.concat([xyz, lg_volume_ratio], dim=1)[:, None, :] # ?
        x_g = self.f_gc(x_g)
        res = self.last_act(torch.concat([x_f + sc, x_g], dim=1))

        return res

class LocalContextLearning(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

        self.local_polar_representation()


class LocalPolarRepresentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def relative_pos_transforming(self, xyz, neigh_idx, neighbor_xyz):
        xyz_tile = torch.tile(xyz[:, :, :, None], [1, 1, 1, neigh_idx.shape[-1]])
        relative_xyz = xyz_tile - neighbor_xyz

        relative_alpha = tf.expand_dims(tf.atan2(relative_xyz[:,:,:,1], relative_xyz[:,:,:,0]), axis=-1)
        relative_xydis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz[:,:,:,:2]), axis=-1))
        relative_beta = tf.expand_dims(tf.atan2(relative_xyz[:,:,:,2], relative_xydis), axis=-1)
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))

        relative_info = tf.concat([relative_dis, xyz_tile, neighbor_xyz], axis=-1)
        
        # negative exp of geometric distance
        exp_dis = tf.exp(-relative_dis)

        # volume of local region 
        local_volume = tf.pow(tf.reduce_max(tf.reduce_max(relative_dis, -1), -1), 3)

        return relative_info, relative_alpha, relative_beta, exp_dis, local_volume

