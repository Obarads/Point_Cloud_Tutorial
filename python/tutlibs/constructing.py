import numpy as np
from .transformation import TransformationMatrix as tm

def marching_cubes():
    return

def tsdf():
    return

def voxel_to_point(voxel:np.ndarray):
    side_len = len(voxel)
    side_idxs = np.arange(side_len)
    x_idxs = np.tile(side_idxs[np.newaxis, np.newaxis, :], (side_len, side_len, 1))
    y_idxs = np.transpose(x_idxs, (1,2,0))
    z_idxs = np.transpose(x_idxs, (2,0,1))
    sides_idxs = np.stack((x_idxs, y_idxs, z_idxs), axis=-1).astype(np.float32)

    point_cloud = sides_idxs[voxel.astype(np.bool8)]
    
    max_xyz = np.max(point_cloud, axis=0)
    min_xyz = np.min(point_cloud, axis=0)

    point_cloud -= ((max_xyz - min_xyz)/2 + min_xyz)

    return point_cloud


def point_to_image(point_cloud:np.ndarray, img:np.ndarray, fx:float, fy:float,
                   cx:float, cy:float, rot_mat:np.ndarray, trans_mat:np.ndarray):
    img_y, img_x, _ = img.shape

    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    rt_mat = np.concatenate([rot_mat, trans_mat[:, np.newaxis]], axis=1)
    p_img = np.matmul(
        np.matmul(intrinsic_matrix, rt_mat),
        np.concatenate([point_cloud, np.full((len(point_cloud), 1), 1)], axis=1).T
    ).T

    # ad = p_img[:, 2][:, np.newaxis]
    p_img = p_img[:, :2] / p_img[:, 2][:, np.newaxis]
    p_img = np.array([[img_x, img_y]]) - p_img[:, :2]

    p_img = np.ceil(p_img)
    img_idxs = p_img.astype(np.int32)

    img_idxs = img_idxs[img_idxs[:, 0] < img_x]
    img_idxs = img_idxs[img_idxs[:, 0] >= 0]
    img_idxs = img_idxs[img_idxs[:, 1] < img_y]
    img_idxs = img_idxs[img_idxs[:, 1] >= 0]
    img[img_idxs[:, 1], img_idxs[:, 0]] = np.array([0, 0, 0], dtype=np.int32)
    

    return img, p_img


def depth_to_point(depth_image:np.ndarray, fx:float, fy:float, cx:float, cy:float, S:float=1):
    img_y, img_x = depth_image.shape
    inverse_intrinsic_matrix = np.array([
        [1/fx, -S/(fx*fy), (S*cy-cx*fy)/(fx*fy), 0],
        [0, 1/fy, -cy/fy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    p_image = np.concatenate(
        [
            np.tile(np.arange(img_x)[:, np.newaxis, np.newaxis], (1, img_y, 1)),
            np.tile(np.arange(img_y)[np.newaxis, :, np.newaxis], (img_x, 1, 1)),
            np.full((img_x, img_y, 2), fill_value=1)
        ],
        axis=2,
    ).astype(np.float32).reshape(-1, 4)

    z = depth_image.T.reshape(-1, 1)
    point_cloud = np.matmul(inverse_intrinsic_matrix, p_image.T).T * z

    return point_cloud[:, :3]

