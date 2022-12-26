from path import add_dir_path
add_dir_path()

import argparse
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from tutlibs.torchlibs.models.PointNet2 import PointNet2SSGClassification
from tutlibs.torchlibs.dataset import (
    ModelNet40DatasetForPointNet,
    rotate_point_cloud,
    jitter_point_cloud,
)


def collate_fn(batch):
    point_clouds, labels = list(zip(*batch))

    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    point_clouds = rotate_point_cloud(point_clouds)
    point_clouds = jitter_point_cloud(point_clouds)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return point_clouds, labels


def test(
    device: int,
    dataset_dir_path: str,
    model_file_path: str,
    num_points: int = 1024,
    num_classes=40,
):
    dataset = ModelNet40DatasetForPointNet(dataset_dir_path, mode="test")

    model = PointNet2SSGClassification(num_classes)
    model = model.to(device=device)
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = DataLoader(dataset, 32, shuffle=False, num_workers=16)
    loader = tqdm(loader, desc="test", ncols=60)

    results = []

    with torch.no_grad():
        for data in loader:
            point_clouds, gt_labels = data

            point_clouds = point_clouds[:, 0:num_points]
            point_clouds = torch.transpose(point_clouds, 1, 2).to(
                device, dtype=torch.float32
            )
            gt_labels = gt_labels.to(device, dtype=torch.long)

            net_output = model(point_clouds, None)
            pred_labels = torch.argmax(net_output, dim=1)
            results.append(pred_labels == gt_labels)

    results = torch.cat(results, dim=0)
    acc = torch.sum(results) / len(results) * 100
    print(f"accuracy: {acc}")


def train(
    device: int,
    output_dir_path: str,
    dataset_dir_path: str,
    num_points: int = 1024,
    num_classes=40,
    epochs=250,
):
    os.makedirs(output_dir_path, exist_ok=True)

    model = PointNet2SSGClassification(num_classes)
    model = model.to(device=device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.7,
    )

    loss_ce = nn.CrossEntropyLoss()

    train_dataset = ModelNet40DatasetForPointNet(dataset_dir_path, mode="train")
    epochs = tqdm(range(epochs), desc="train", ncols=60)

    for epoch in epochs:
        # train
        loader = DataLoader(
            train_dataset,
            32,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=16,
        )
        model.train()
        for data in loader:
            optimizer.zero_grad()

            point_clouds, gt_labels = data

            point_clouds = point_clouds[:, 0:num_points]
            point_clouds = point_clouds.to(device=device).transpose(1, 2)
            gt_labels = gt_labels.to(device=device)

            net_output = model(point_clouds, None)
            loss = loss_ce(net_output, gt_labels)

            loss.backward()
            optimizer.step()

        scheduler.step()

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(output_dir_path, "model.pth"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="select `train` or `test` process",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="gpu device number",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="outputs/pointnet2/",
    )
    parser.add_argument(
        "--test_model_file_path",
        type=str,
        default="outputs/pointnet2/model.pth",
        help="trained param file (model.pth) path, only mode=test",
    )
    parser.add_argument(
        "--dataset_dir_path",
        type=str,
        default="../../data/modelnet40_ply_hdf5_2048/",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.device, args.output_dir_path, args.dataset_dir_path)
    elif args.mode == "test":
        test(args.device, args.dataset_dir_path, args.test_model_file_path)
    else:
        raise ValueError()
