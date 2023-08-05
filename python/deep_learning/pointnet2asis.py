from path import add_dir_path

add_dir_path()

import argparse
from tqdm import tqdm
import numpy as np
import os
from sklearn.cluster import MeanShift

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from typing import Dict, List, Tuple

from tutlibs.torchlibs.models.PointNet2ASIS import PointNet2ASIS
from tutlibs.torchlibs.dataset.s3dis import S3DISDatasetForASIS
from tutlibs.torchlibs.dataset.augmentation import (
    rotate_point_cloud,
    jitter_point_cloud,
)
from tutlibs.torchlibs.dataset.postprocessing import BlockMerging
from tutlibs.torchlibs.loss import DiscriminativeLoss
from tutlibs.torchlibs.utils import torch_seed
from tutlibs.utils import env_seed
from tutlibs.metric import SemanticMetrics, CoverageMetrics


def collate_fn(batch):
    point_clouds, labels = list(zip(*batch))

    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    point_clouds = rotate_point_cloud(point_clouds)
    point_clouds = jitter_point_cloud(point_clouds)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return point_clouds, labels


def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    ms.fit(prediction)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers


def test(
    device: int,
    dataset_dir_path: str,
    model_file_path: str,
):
    dataset = S3DISDatasetForASIS(
        dataset_dir_path, mode="test", test_area_number=5
    )

    model = PointNet2ASIS(dataset.num_classes)
    model = model.to(device=device)
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # loader = DataLoader(dataset, 24, shuffle=False, num_workers=16)
    # loader = tqdm(loader, desc="test", ncols=60)

    room_names = dataset.get_room_names()
    # results_per_room: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    coverage_mertrics = CoverageMetrics(dataset.num_classes)
    semantic_mertrics = SemanticMetrics(dataset.num_classes)

    with torch.no_grad():
        for room_name in room_names:
            block_merging = BlockMerging(room_name)
            start_idx, end_idx = dataset.block_index_range_per_room(room_name)
            sem_prediction_label_list = []
            ins_prediction_label_list = []
            sem_gt_label_list = []
            ins_gt_label_list = []
            for idx in range(start_idx, end_idx):
                point_clouds, semantic_labels, instance_labels = dataset[idx]

                input_point_clouds = torch.transpose(point_clouds, 0, 1).to(
                    device, dtype=torch.float32
                )
                coords = input_point_clouds[None, 0:3, :]
                features = input_point_clouds[None, 3:, :]
                semantic_labels = semantic_labels.to(device, dtype=torch.long)
                instance_labels = instance_labels.to(device, dtype=torch.long)

                sem_net_output, ins_net_output = model(coords, features)

                sem_prediction_labels = torch.argmax(sem_net_output, dim=1)

                ins_prediction_labels = cluster(ins_net_output, 1)[1]
                ins_prediction_labels = block_merging.assign(
                    point_clouds[:, 6:],
                    ins_prediction_labels,
                    sem_prediction_labels,
                )

                sem_prediction_label_list.append(sem_prediction_labels)
                sem_gt_label_list.append(semantic_labels)
                ins_prediction_label_list.append(ins_prediction_labels)
                ins_gt_label_list.append(instance_labels)

            coverage_mertrics.update(
                np.concatenate(sem_prediction_label_list),
                np.concatenate(ins_prediction_label_list),
                np.concatenate(sem_gt_label_list),
                np.concatenate(ins_gt_label_list),
            )
            semantic_mertrics.update(sem_prediction_label_list, sem_gt_label_list)

    print("Instance Metrics")
    print(f"Cov: {coverage_mertrics.mucov()}")
    print(f"wCov: {coverage_mertrics.mwcov()}")
    print(f"IoU: {semantic_mertrics.iou()}")

    print("Semantic Metrics")
    print(f"Accuracy: {semantic_mertrics.accuracy()}")
    print(f"Precision: {semantic_mertrics.precision()}")
    print(f"Recall: {semantic_mertrics.recall()}")
    print(f"f1: {semantic_mertrics.f1()}")


def train(
    device: int,
    output_dir_path: str,
    dataset_dir_path: str,
    num_classes=40,
    epochs=250,
):
    os.makedirs(output_dir_path, exist_ok=True)

    model = PointNet2ASIS(num_classes, 5)
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

    sem_cel = nn.CrossEntropyLoss()
    ins_dl = DiscriminativeLoss(1.5, 0.5, 1.0, 1.0, 0.001)

    train_dataset = S3DISDatasetForASIS(
        dataset_dir_path, mode="train", test_area_number=5
    )
    epochs = tqdm(range(epochs), desc="train", ncols=60)

    for epoch in epochs:
        # train
        loader = DataLoader(
            train_dataset,
            24,
            shuffle=True,
            # collate_fn=collate_fn,
            num_workers=16,
        )
        model.train()

        for data in loader:
            optimizer.zero_grad()

            point_clouds, semantic_labels, instance_labels = data

            point_clouds = torch.transpose(point_clouds, 1, 2).to(
                device, dtype=torch.float32
            )
            coords = point_clouds[:, 0:3, :]
            features = point_clouds[:, 3:, :]
            semantic_labels = semantic_labels.to(device, dtype=torch.long)
            instance_labels = instance_labels.to(device, dtype=torch.long)

            sem_net_output, ins_net_output = model(coords, features)
            sem_loss = sem_cel(sem_net_output, semantic_labels)
            ins_loss = ins_dl(ins_net_output, instance_labels)
            loss = sem_loss + ins_loss

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
        default="outputs/pointnet2asis/",
    )
    parser.add_argument(
        "--test_model_file_path",
        type=str,
        default="outputs/pointnet2asis/model.pth",
        help="trained param file (model.pth) path, only mode=test",
    )
    parser.add_argument(
        "--dataset_dir_path",
        type=str,
        default="../../data/s3dis/preprocessed_data/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    torch_seed(args.seed)
    env_seed(args.seed)

    if args.mode == "train":
        train(args.device, args.output_dir_path, args.dataset_dir_path)
    elif args.mode == "test":
        test(args.device, args.dataset_dir_path, args.test_model_file_path)
    else:
        raise ValueError()
