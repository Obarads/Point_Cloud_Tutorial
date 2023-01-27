import numpy as np
import pandas as pd
import os, sys, glob, pickle
from sklearn.neighbors import KDTree
import argparse
from typing import List, Tuple

sys.path.append("../")

from libs.io import ply
from libs.logs import LoggingUtils, save_args

# from helper_dp import DataProcessing as DP

logger = LoggingUtils.create_logger(os.path.basename(__file__))


def create_original_point_data_files(
    point_data, output_dir_path, filename_wo_ext
):
    ply.write(
        os.path.join(output_dir_path, f"{filename_wo_ext}.ply"), point_data
    )

    


def convert_point_data_to_preprocessed_files(anno_path, save_path):
    data_list = []

    for f in glob.glob(join(anno_path, "*.txt")):
        class_name = os.path.basename(f).split("_")[0]
        if (
            class_name not in gt_class
        ):  # note: in some room there is 'staris' class..
            class_name = "clutter"
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    write_ply(
        save_path,
        (xyz, colors, labels),
        ["x", "y", "z", "red", "green", "blue", "class"],
    )

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(
        xyz, colors, labels, sub_grid_size
    )
    sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, save_path.split("/")[-1][:-4] + ".ply")
    write_ply(
        sub_ply_file,
        [sub_xyz, sub_colors, sub_labels],
        ["x", "y", "z", "red", "green", "blue", "class"],
    )

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(
        sub_pc_folder, str(save_path.split("/")[-1][:-4]) + "_KDTree.pkl"
    )
    with open(kd_tree_file, "wb") as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(
        sub_pc_folder, str(save_path.split("/")[-1][:-4]) + "_proj.pkl"
    )
    with open(proj_save, "wb") as f:
        pickle.dump([proj_idx, labels], f)


def load_s3dis(
    data_dir_path: str, align_to_origin: bool = True
) -> Tuple(List[np.ndarray], List[Tuple[str, str]], List[str]):
    """Load S3DIS data.

    Args:
        data_dir_path: a path to Stanford3dDataset_v1.2_Aligned_Version
        align_to_origin: align point clouds to origin

    Return:
        room_point_data_list: each room data list, dtype is room_point_type.
        room_point_type: fields and type of room_point_data_list.
        room_paths: each S3DIS room dir path
    """
    label_to_class_dict = {
        0: "ceiling",
        1: "floor",
        2: "wall",
        3: "beam",
        4: "column",
        5: "window",
        6: "door",
        7: "table",
        8: "chair",
        9: "sofa",
        10: "bookcase",
        11: "board",
        12: "clutter",
    }
    class_to_label_dict = {
        label_to_class_dict[key]: key for key in label_to_class_dict
    }
    class_converter = {
        "staris": "clutter",
    }
    point_type = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "i4"),
        ("green", "i4"),
        ("blue", "i4"),
    ]
    room_point_type = point_type + [("semantic_label", "i4")]

    room_point_data_list = []
    room_paths = sorted(glob(os.path.join(data_dir_path, "*/*/")))
    for room_path in room_paths:
        room_annotation_dir_path = os.path.join(room_path, "Annotations")
        annotation_file_paths = sorted(
            glob.glob(os.path.join(room_annotation_dir_path, "*.txt"))
        )

        points_list = []
        instance_id = 0
        for annotation_file_path in annotation_file_paths:
            class_name = os.path.basename(annotation_file_path).split("_")[0]
            if class_name not in class_converter:
                class_name = class_converter[class_name]

            if class_name in class_to_label_dict:
                points = np.loadtxt(
                    annotation_file_path,
                    dtype=point_type,
                )

                if align_to_origin:
                    for field_name in ["x", "y", "z"]:
                        points[field_name] -= np.min(points[field_name])

                semantic_labels = np.full(
                    len(points),
                    fill_value=class_to_label_dict[class_name],
                    dtype=np.int32,
                )
                points_list.append(
                    np.stack([points, semantic_labels], axis=-1)  # [N, 7]
                )

                instance_id += 1
            else:
                raise ValueError()

        point_data = np.concatenate(
            points_list,
            axis=0,
            dtype=room_point_type,
        )
        room_point_data_list.append(point_data)

    return room_point_data_list, room_point_type, room_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_path",
        "-i",
        type=str,
        default="./data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version",
    )
    parser.add_argument(
        "--output_dir_path",
        "-o",
        type=str,
        default="data/S3DIS/preprocessed_data",
    )
    parser.add_argument("--sub_grid_size", "-s", type=str, default=0.04)
    args = parser.parse_args()

    output_dir_path = args.output_dir_path
    data_dir_path = args.data_dir_path
    sub_grid_size = args.sub_grid_size

    log_dir_path = os.path.join(args.output_dir_path, "log")
    os.makedirs(log_dir_path, exist_ok=True)
    save_args(os.path.join(log_dir_path, "args.yaml"), args)
    LoggingUtils.add_file_handler(
        os.path.join(log_dir_path, "logger.log"), logger
    )

    logger.info("Load S3DIS data")
    room_point_data_list, point_type, data_paths = load_s3dis(data_dir_path)

    logger.info("Preprocess each room data")
    for point_data, data_path in zip(room_point_data_list, data_paths):
        logger.info(f"Preprocess data: {data_path}")

        filename_wo_ext = os.path.splitext(os.path.basename(data_path))[0]
        original_ply_dir_path = os.path.join(output_dir_path, "original_ply")
        create_original_point_data_files(
            point_data, original_ply_dir_path, filename_wo_ext
        )
        convert_point_data_to_preprocessed_files(point_data, filename_wo_ext)

    original_pc_folder = join(dirname(dataset_path), "original_ply")
    sub_pc_folder = join(
        dirname(dataset_path), "input_{:.3f}".format(sub_grid_size)
    )
    os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
    os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
    out_format = ".ply"

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split("/")
        out_file_name = elements[-3] + "_" + elements[-2] + out_format
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name))
