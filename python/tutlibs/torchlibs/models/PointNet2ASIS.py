import torch
from torch import nn

from tutlibs.torchlibs.models.PointNet2 import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
    MLP,
)
from tutlibs.torchlibs.nns import k_nearest_neighbors
from tutlibs.torchlibs.operator import index2points


class ASIS(nn.Module):
    def __init__(
        self,
        sem_in_channels,
        sem_out_channels,
        ins_in_channels,
        ins_out_channels,
        k,
        memory_saving=True,
    ):
        super(ASIS, self).__init__()

        # sem branch
        self.sem_pred_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(sem_in_channels, sem_out_channels, 1),
        )  # input: F_ISEM, output: P_SEM

        # interactive module: sem to ins
        self.adaptation = MLP(sem_in_channels, ins_in_channels, conv_dim=1)

        # ins branch
        self.ins_emb_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(ins_in_channels, ins_out_channels, 1),
        )  # input: F_SINS, output: E_INS

        # interactive module: ins to sem
        # using knn_index and index2points

        self.k = k
        self.memory_saving = memory_saving

    def forward(self, f_sem, f_ins):
        adapted_f_sem = self.adaptation(f_sem)

        # for E_INS
        f_sins = f_ins + adapted_f_sem
        e_ins = self.ins_emb_fc(f_sins)

        # for P_SEM
        nn_idx, _ = k_nearest_neighbors(e_ins, e_ins, self.k)
        k_f_sem = index2points(f_sem, nn_idx)
        f_isem = torch.max(k_f_sem, dim=3, keepdim=True)[0]
        f_isem = torch.squeeze(f_isem, dim=3)
        p_sem = self.sem_pred_fc(f_isem)

        return p_sem, e_ins


class PointNet2ASIS(nn.Module):
    def __init__(self, num_classes, ins_channel):
        super(PointNet2ASIS, self).__init__()

        self.sa1 = PointNetSetAbstraction(
            1024, 0.1, 32, 6 + 3, [32, 32, 64], None
        )
        self.sa2 = PointNetSetAbstraction(
            256, 0.2, 32, 64 + 3, [64, 64, 128], None
        )
        self.sa3 = PointNetSetAbstraction(
            64, 0.4, 32, 128 + 3, [128, 128, 256], None
        )
        self.sa4 = PointNetSetAbstraction(
            16, 0.8, 32, 256 + 3, [256, 256, 512], None
        )

        self.sem_fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.sem_fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.sem_fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.sem_fp1 = PointNetFeaturePropagation(128 + 6, [128, 128, 128])

        self.ins_fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.ins_fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.ins_fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.ins_fp1 = PointNetFeaturePropagation(128 + 6, [128, 128, 128])

        self.sem_fc = MLP(128, 128, conv_dim=1)  # for F_SEM
        self.ins_fc = MLP(128, 128, conv_dim=1)  # for F_INS

        self.asis = ASIS(128, num_classes, 128, ins_channel, 30)

    def forward(self, coords: torch.tensor, features: torch.tensor = None):
        l1_coords, l1_features = self.sa1(coords, features)
        l2_coords, l2_features = self.sa2(l1_coords, l1_features)
        l3_coords, l3_features = self.sa3(l2_coords, l2_features)
        l4_coords, l4_features = self.sa4(l3_coords, l3_features)

        l3_sem_features = self.sem_fp4(
            l3_coords, l4_coords, l3_features, l4_features
        )
        l2_sem_features = self.sem_fp3(
            l2_coords, l3_coords, l2_features, l3_sem_features
        )
        l1_sem_features = self.sem_fp2(
            l1_coords, l2_coords, l1_features, l2_sem_features
        )
        l0_sem_features = self.sem_fp1(
            coords, l1_coords, features, l1_sem_features
        )

        l3_ins_features = self.ins_fp4(
            l3_coords, l4_coords, l3_features, l4_features
        )
        l2_ins_features = self.ins_fp3(
            l2_coords, l3_coords, l2_features, l3_ins_features
        )
        l1_ins_features = self.ins_fp2(
            l1_coords, l2_coords, l1_features, l2_ins_features
        )
        l0_ins_features = self.ins_fp1(
            coords, l1_coords, features, l1_ins_features
        )

        f_sem = self.sem_fc(l0_sem_features)
        f_ins = self.ins_fc(l0_ins_features)

        p_sem, e_ins = self.asis(f_sem, f_ins)

        return p_sem, e_ins
