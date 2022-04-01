import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from tutlibs.dl.PointNet import PointNetClassification
from tutlibs.dl.dataset import (
    ModelNet40Dataset,
    rotate_point_cloud,
    jitter_point_cloud,
)
from tutlibs.dl.loss import feature_transform_regularizer
from tutlibs.dl.utils import t2n

def test():
    device = 0
    output_dir_path = "outputs/PointNet/"
    dataset_dir_path = "../data/modelnet40_ply_hdf5_2048/"
    num_points = 1024
    num_classes = 40

    os.makedirs(output_dir_path, exist_ok=True)

    dataset = ModelNet40Dataset(dataset_dir_path, mode="test")

    model = PointNetClassification(num_classes)
    model = model.to(device=device)
    checkpoint = torch.load(os.path.join(output_dir_path, "model_path.pth"))
    model.load_state_dict(checkpoint["model"])

    loader = DataLoader(dataset, 32, shuffle=False)
    model.eval()

    results = []

    with torch.no_grad():
        for data in tqdm(loader, desc="test", ncols=60):
            point_clouds, gt_labels = data

            point_clouds = point_clouds[:, 0:num_points]
            point_clouds = torch.transpose(point_clouds, 1, 2).to(
                device, dtype=torch.float32
            )
            gt_labels = gt_labels.to(device, dtype=torch.long)

            net_output, _, _ = model(point_clouds)
            pred_labels = torch.argmax(net_output, dim=1)
            results.append(pred_labels == gt_labels)

    results = torch.cat(results, dim=0)
    acc = torch.sum(results) / len(results) * 100
    print(f"accuracy: {acc}")


test()

