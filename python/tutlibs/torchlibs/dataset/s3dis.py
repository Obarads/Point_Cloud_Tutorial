from glob import glob
import os
import numpy as np
import h5py
from typing import Tuple, List, Dict

from numpy.lib import recfunctions
from torch.utils.data import Dataset


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
    "stairs": "clutter",
}


def load_room_data(room_path: str, align_to_origin: bool = True):
    r"""Load S3DIS room data with a room directory path

    Args:
        room_path: string path of room directory in S3DIS Area_X directory
        align_to_origin: A boolean indicating if we align the minimum
            coordinates of room points with the originthe room position.

    Returns:
        Array: The room point cloud including N points. A point in the point
            cloud have [x, y, z, red, green. blue, semantic_label,
            instance_label] fields.

    Examples:
        >>> path = "Stanford3dDataset_v1.2_Aligned_Version/Area_1/office_1"
        >>> point_data = load_room_data(path)
        >>> point_data["x"].shape
        (884955,)
    """

    room_annotation_dir_path = os.path.join(room_path, "Annotations")
    annotation_file_paths = sorted(
        glob(os.path.join(room_annotation_dir_path, "*.txt"))
    )

    points_list = []
    point_type = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "i4"),
        ("green", "i4"),
        ("blue", "i4"),
    ]
    instance_id = 0
    for annotation_file_path in annotation_file_paths:
        class_name = os.path.basename(annotation_file_path).split("_")[0]
        if class_name in class_converter:
            class_name = class_converter[class_name]

        if class_name in class_to_label_dict:
            points = np.loadtxt(
                annotation_file_path,
                dtype=point_type,
            )

            if align_to_origin:
                for field_name in ["x", "y", "z"]:
                    points[field_name] -= np.min(points[field_name])

            points = recfunctions.append_fields(
                points,
                "semantic_label",
                np.full(
                    len(points),
                    fill_value=class_to_label_dict[class_name],
                    dtype=np.int32,
                ),
                usemask=False,
                asrecarray=False,
            )  # [N, 7]
            points = recfunctions.append_fields(
                points,
                "instance_label",
                np.full(len(points), fill_value=instance_id, dtype=np.int32),
                usemask=False,
                asrecarray=False,
            )  # [N, 8]
            points_list.append(points)

            instance_id += 1
        else:
            raise ValueError()

    point_data = np.concatenate(points_list, axis=0)

    return point_data


def room_to_block(
    point_data: str, block_size: float = 1.0, stride: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Create masks of points for blocks

    Args:
        room_path: Point data including ["x", "y", "z"] fields
        block_size: block size
        stride: stride

    Returns:
        Boolean mask array: The array (shape: [number of blocks, number of
            points in the room]) indicates points that a block includes.
            If True is a point in the room, the point belongs to the block.
            (ex: block_index_list[2, 50] is True, i.e., the 50th point in the
            room point cloud is contained in the second block.)
        Float array: The array (shape: [number of blocks, 4]) indicates xy
            coordinates of corner of each block. 4 elements is [min x, min y,
            max x, max y].
    """

    # Compute local xyz coordinates
    coords = np.stack(
        [point_data["x"], point_data["y"], point_data["z"]], axis=-1
    )
    coords = coords - np.min(coords, axis=0)  # [N, 3]

    # Create Block indices
    max_coords = np.max(coords, axis=0)
    num_xy_block = (
        np.ceil((max_coords[:2] - block_size) / stride).astype(np.int32) + 1
    )
    block_index_list = np.stack(
        [
            np.tile(
                np.arange(num_xy_block[0])[:, np.newaxis],
                (1, num_xy_block[1]),
            ),
            np.tile(
                np.arange(num_xy_block[1])[np.newaxis, :],
                (num_xy_block[0], 1),
            ),
        ],
        axis=-1,
    ).reshape(
        -1, 2
    )  # [number of blocks, 2], 2=block index with x and y axis

    # Create masks of each block on room point cloud
    block_mask_list = []
    block_corner_list = []
    xy_coords = coords[:, :2]
    for block_index in block_index_list:
        block_max_xy_coord = block_index[np.newaxis, :] * stride + block_size
        block_min_xy_coord = block_index[np.newaxis, :] * stride
        xy_mask = (xy_coords <= block_max_xy_coord) & (
            xy_coords >= block_min_xy_coord
        )
        mask = xy_mask[:, 0] & xy_mask[:, 1]

        block_mask_list.append(mask)
        block_corner_list.append(
            np.concatenate(
                [block_min_xy_coord.flatten(), block_max_xy_coord.flatten()]
            )
        )

    # [number of blocks, number of points in the room]
    # If True is a point in the room, the point belongs to the block.
    # (ex: block_index_list[2, 50] is True, i.e., the 50th point in the room
    # point cloud is contained in the second block.)
    block_index_list = np.array(block_index_list, dtype=bool)

    # Each block XY corner, this array is used in normalization for XY
    # coordinates.
    block_corner_list = np.array(block_corner_list, dtype=np.float32)

    return block_mask_list, block_corner_list


def remove_blocks(block_mask_list: np.ndarray, threshold: int):
    r"""Remove blocks of points below threshold.

    Args:
        block_mask_list: Masks indicating points that a block includes
            (shape: [number of blocks, number of points in the room])
        threshold: Threshold to remove blocks of points below any number

    Returns:
        boolean mask array: Masks of block with points above threshold
    """
    block_point_counts = np.sum(block_mask_list, axis=1)
    mask_to_keep_blocks = block_point_counts > threshold
    block_mask_list = block_mask_list[mask_to_keep_blocks]
    return block_mask_list


def create_block_points(
    block_mask_list: np.ndarray,
    room_points: np.ndarray,
    num_sample_points: int = 4096,
):
    r"""Create blocks.

    Args:
        block_mask_list: Masks indicating points that a block includes
            (shape: [number of blocks, number of points in the room])
        room_points: point data in the room (shape: [number of points in the
            room])
        num_sample_points: number of points in a block

    Returns:
        Array: blocks (shape: [number of blocks, num_sample_points])
    """
    point_type = room_points.dtype
    sampled_block_points = np.full(
        (len(block_mask_list), num_sample_points),
        fill_value=-1,
        dtype=point_type,
    )

    for block_index, block_mask in enumerate(block_mask_list):
        block_points = room_points[block_mask]
        num_block_points = len(block_points)
        if num_block_points >= num_sample_points:
            block_point_indices = np.random.choice(
                num_block_points, num_sample_points, replace=False
            )
        else:
            block_point_indices = np.random.choice(
                num_block_points, num_sample_points, replace=True
            )

        sampled_block_points[block_index] = block_points[block_point_indices]

    return sampled_block_points


def create_and_normalize_point_features(
    block_point_clouds: np.ndarray,
    room_points: np.ndarray,
    block_size: float,
):
    """Create and normalize point features

    Args:
        block_point_clouds: blocks (shape: [number of blocks, number of points])
        room_points: points in the room (shape: [number of points in the room])
        block_size: block size

    Returns:
        Array: The block point clouds including N points. A point in the point
            cloud have [normalized_room_x, normalized_room_y, normalized_room_z,
            normalized_red, normalized_green. normalized_blue, block_x, block_y,
            block_z] fields and other room_points fields.
    """
    point_type = [
        ("normalized_room_x", "f4"),
        ("normalized_room_y", "f4"),
        ("normalized_room_z", "f4"),
        ("normalized_red", "f4"),
        ("normalized_green", "f4"),
        ("normalized_blue", "f4"),
        ("block_x", "f4"),
        ("block_y", "f4"),
        ("block_z", "f4"),
    ]
    num_blocks, num_points = block_point_clouds.shape
    new_block_point_clouds = np.zeros(
        (num_blocks, num_points), dtype=point_type
    )

    new_block_point_clouds["normalized_red"] = block_point_clouds["red"] / 255
    new_block_point_clouds["normalized_green"] = (
        block_point_clouds["green"] / 255
    )
    new_block_point_clouds["normalized_blue"] = block_point_clouds["blue"] / 255

    new_block_point_clouds["normalized_room_x"] = (
        block_point_clouds["x"] - np.min(room_points["x"])
    ) / (np.max(room_points["x"]) - np.min(room_points["x"]))
    new_block_point_clouds["normalized_room_y"] = (
        block_point_clouds["y"] - np.min(room_points["y"])
    ) / (np.max(room_points["y"]) - np.min(room_points["y"]))
    new_block_point_clouds["normalized_room_z"] = (
        block_point_clouds["z"] - np.min(room_points["z"])
    ) / (np.max(room_points["z"]) - np.min(room_points["z"]))

    new_block_point_clouds["block_x"] = (
        block_point_clouds["x"]
        - np.min(block_point_clouds["x"], axis=1, keepdims=True)
        + block_size / 2
    )
    new_block_point_clouds["block_y"] = (
        block_point_clouds["y"]
        - np.min(block_point_clouds["y"], axis=1, keepdims=True)
        + block_size / 2
    )
    new_block_point_clouds["block_z"] = block_point_clouds["z"]

    return new_block_point_clouds


def concat_field(
    base: np.ndarray, additional_array: np.ndarray, field_names: List[str]
):
    """Concatenate fields of two arrays along field_names.

    Args:
        base: base array
        additional_array: additional array
        field_names: field names to concatenate addtional_array to base

    Return:
        Array: concatenated base array
    """
    # base = np.copy(base)
    # for field_name in field_names:
    #     base = recfunctions.append_fields(
    #         base,
    #         field_name,
    #         additional_array[field_name],
    #         usemask=False,
    #         asrecarray=False,
    #     )
    new_fields = []
    for n in base.dtype.names:
        new_fields += [(n, base[n].dtype)]
    for n in field_names:
        new_fields += [(n, additional_array[n].dtype)]

    new_array = np.zeros(base.shape, dtype=new_fields)

    for n in base.dtype.names:
        new_array[n] = base[n]
    for n in field_names:
        new_array[n] = additional_array[n]

    return new_array


def preprocess_s3dis(
    s3dis_dir_path: str,
    output_dir_path: str,
    block_size: float,
    stride: float,
    num_sampled_points: int,
    threshold: int,
):
    """Create blocks of area block_size^2 (along the ground) from raw S3DIS
    dataset.
    Also, when create blocks, sample num_sampled_points points from each block.
    This method follows the preprocessing method of PointNet and ASIS.

    Args:
        s3dis_dir_path: path to the directory of S3DIS dataset
        output_dir_path: path to the directory to save the preprocessed data
        block_size: block size
        stride: stride
        num_sampled_points: number of sampled points
        theshold: remove blocks with less than threshold points

    Returns:
        None
    """
    os.makedirs(output_dir_path, exist_ok=True)

    room_paths = sorted(glob(os.path.join(s3dis_dir_path, "*/*/")))
    for room_path in room_paths:

        room_point_data = load_room_data(room_path, align_to_origin=True)
        block_mask_list, _ = room_to_block(room_point_data, block_size, stride)
        block_mask_list = np.array(block_mask_list)
        block_mask_list = remove_blocks(block_mask_list, threshold)
        block_point_clouds = create_block_points(
            block_mask_list, room_point_data, num_sampled_points
        )
        block_features = create_and_normalize_point_features(
            block_point_clouds, room_point_data, block_size
        )
        block_point_clouds = concat_field(
            block_features,
            block_point_clouds,
            ["semantic_label", "instance_label"],
        )

        path_info = room_path.rstrip("/").split("/")
        area_name = path_info[-2]
        room_name = path_info[-1]
        with h5py.File(
            os.path.join(output_dir_path, f"{area_name}_{room_name}.h5"), "w"
        ) as f:
            for field_name in block_point_clouds.dtype.names:
                point_field = block_point_clouds[field_name]
                f.create_dataset(field_name, data=point_field)


class S3DISDatasetForASIS(Dataset):
    """S3DIS dataset for ASIS.

    Args:
        preprocessed_data_dir_path: path to the directory of preprocessed data
        mode: train or test
        test_area_number: test area number (1-6)
    """

    point_type = [
        ("normalized_room_x", "f4"),
        ("normalized_room_y", "f4"),
        ("normalized_room_z", "f4"),
        ("normalized_red", "f4"),
        ("normalized_green", "f4"),
        ("normalized_blue", "f4"),
        ("block_x", "f4"),
        ("block_y", "f4"),
        ("block_z", "f4"),
        ("semantic_label", "i4"),
        ("instance_label", "i4"),
    ]
    num_classes = len(label_to_class_dict)

    def __init__(
        self,
        preprocessed_data_dir_path: str,
        mode: str,
        test_area_number: int = 5,
    ) -> None:
        super().__init__()

        assert mode in ["train", "test"]
        assert test_area_number in [i for i in range(1, 7)]

        self.preprocessed_data_dir_path = preprocessed_data_dir_path
        self.mode = mode
        self.data: List[np.ndarray] = []
        self.test_area_number = test_area_number
        self.load_h5()

    def load_h5(self):
        """Load h5 files (preprocessed data)."""
        if self.mode == "train":
            glob_path = os.path.join(
                self.preprocessed_data_dir_path,
                f"Area_[!{self.test_area_number}]_*.h5",
            )
        elif self.mode == "test":
            glob_path = os.path.join(
                self.preprocessed_data_dir_path,
                f"Area_{self.test_area_number}_*.h5",
            )
        else:
            raise NotImplementedError()

        self.preprocessed_data_file_paths = sorted(glob(glob_path))
        self.data: List[np.ndarray] = []
        self.block_index_range_pre_room_dict: Dict[str, Tuple[int, int]] = {}
        current_idx = 0
        for preprocessed_data_file_path in self.preprocessed_data_file_paths:
            with h5py.File(preprocessed_data_file_path, "r") as f:
                data = np.zeros(
                    f[list(f.keys())[0]].shape, dtype=self.point_type
                )

                for field_name in data.dtype.names:
                    data[field_name] = f[field_name]

            self.data.append(data)

            previous_idx = current_idx
            current_idx += len(data)
            self.block_index_range_pre_room_dict[
                os.path.basename(preprocessed_data_file_path)
            ] = (previous_idx, current_idx)
        
        self.data = np.concatenate(self.data, axis=0)

    def block_index_range_per_room(self, room_name: str) -> Tuple[int, int]:
        """Get the index range of blocks in a room.

        Args:
            room_name: room name

        Returns:
            start_index: start index
            end_index: end index
        """

        start_index = 0
        end_index = 0
        if room_name in self.block_index_range_pre_room_dict:
            start_index, end_index = self.block_index_range_pre_room_dict[
                room_name
            ]
        else:
            start_index = None
            end_index = None

        return start_index, end_index

    def get_room_names(self) -> List[str]:
        """Get room names.

        Returns:
            room_names: room names
        """
        return list(self.block_index_range_pre_room_dict.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """__getitem__ method for PyTorch Dataset.

        Args:
            idx: index

        Returns:
            point_cloud: point cloud [num_points, 9]
            semantic_labels: semantic labels [num_points]
            instance_labels: instance labels [num_points]
        """
        data = self.data[idx]
        point_cloud = np.stack(
            [
                data["normalized_room_x"],
                data["normalized_room_y"],
                data["normalized_room_z"],
                data["normalized_red"],
                data["normalized_green"],
                data["normalized_blue"],
                data["block_x"],
                data["block_y"],
                data["block_z"],
            ],
            dtype=np.float32,
            axis=-1,
        )
        semantic_labels = data["semantic_label"]
        instance_labels = data["instance_label"]
        return point_cloud, semantic_labels, instance_labels
