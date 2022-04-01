import os
from tqdm import tqdm
import numpy as np

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
from tutlibs.io import Points


def collate_fn(batch):
    point_clouds, labels = list(zip(*batch))

    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    point_clouds = rotate_point_cloud(point_clouds)
    point_clouds = jitter_point_cloud(point_clouds)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return point_clouds, labels


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


def main():
    epochs = 250
    device = 0
    output_dir_path = "outputs/PointNet"
    dataset_dir_path = "../data/modelnet40_ply_hdf5_2048/"
    num_points = 1024
    num_classes = 40

    os.makedirs(output_dir_path, exist_ok=True)

    train_dataset = ModelNet40Dataset(dataset_dir_path, mode="train")
    test_dataset = ModelNet40Dataset(dataset_dir_path, mode="test")

    model = PointNetClassification(num_classes)
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.7,
    )

    loss_ce = nn.CrossEntropyLoss()
    loss_ftr = feature_transform_regularizer

    for epoch in range(epochs):
        # train
        loader = DataLoader(train_dataset, 32, shuffle=True, collate_fn=collate_fn)
        model.train()
        for data in loader:
            optimizer.zero_grad()

            point_clouds, gt_labels = data

            # Points.write("outputs/test.ply",t2n(point_clouds[0]))

            point_clouds = point_clouds[:, 0:num_points]
            point_clouds = point_clouds.to(device=device).transpose(1, 2)
            gt_labels = gt_labels.to(device=device)

            net_output, _, feature_transformation_matrix = model(point_clouds)

            loss = loss_ce(net_output, gt_labels)
            if feature_transformation_matrix is not None:
                loss += loss_ftr(feature_transformation_matrix) * 0.001

            loss.backward()
            optimizer.step()

        scheduler.step()


        # test
        loader = DataLoader(test_dataset, 32, shuffle=False)
        model.eval()
        results = []
        with torch.no_grad():
            for data in loader:
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
        print(f"Epoch: {epoch}/{epochs}, accuracy: {acc}")


    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(output_dir_path, "model_path.pth"),
    )

main()