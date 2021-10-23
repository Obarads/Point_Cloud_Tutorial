import numpy as np

from tutlibs.registration import icp_ransac
from tutlibs.io import Points as io
from tutlibs.visualization import Points as visualizer
from tutlibs.sampling import voxel_grid_sampling
from tutlibs.transformation import rotation, translation
from tutlibs.transformation import TransformationMatrix as tm

# data acquisition
target_xyz, _, _ = io.read('../data/bunny_pc.ply')

# source random transformation
source_xyz = rotation(translation(target_xyz, np.array([0.0, 0.1, 0.0])), 'z', angle=30/180*np.pi)

# # keypoints estimation
source_v_xyz = voxel_grid_sampling(source_xyz, 0.01)
target_v_xyz = voxel_grid_sampling(target_xyz, 0.01)

# use ICP
trans_mat = icp_ransac(source_v_xyz, target_v_xyz, 20)
estimated_source_xyz = tm.transformation_Nx3_with_4x4(source_xyz, trans_mat)

# visualize results
source_colors = np.tile([[1, 0, 0]], (source_xyz.shape[0], 1))
estimated_source_colors = np.tile([[0, 1, 0]], (estimated_source_xyz.shape[0], 1))
target_colors = np.tile([[1, 1, 0]], (target_xyz.shape[0], 1))

vis_xyz = np.concatenate([estimated_source_xyz, target_xyz, source_xyz])
vis_color = np.concatenate([estimated_source_colors, target_colors, source_colors])
# visualizer.k3d(vis_xyz, colors=vis_color)
# io.write('outputs/r1.ply', vis_xyz, vis_color)


