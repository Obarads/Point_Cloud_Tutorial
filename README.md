# Point cloud tutorial
This repository contains tutorial code and supplementary note for point cloud processing. In order to understand algorithm of point cloud processing, most codes are implemented with [Numpy](https://numpy.org/) and [Jupyter](https://jupyter.org/).

## Repository structure
```bash
┌─ data             # 3D files for examples
├─ .devcontainer    # Dockerfile for this tutorial
└─ python           # python codes of tutorial
    ├─ tutlibs      # package for tutorial codes
    └─ *.ipynb      # tutorial codes
```

## How to use
### 1. Enviroment
You can execute most tutorial codes on [Codespaces](https://github.com/features/codespaces)(CPU resource). If you have GPU resource, can execute all tutorial codes. For more environment of CPU and GPU, Please refer to [.devcontainer/README.md](.devcontainer/README.md).

### 2. About tutorial
Tutorial contents are as follows:

| Theme                                | Page & code                                    | Contents                                                                                     | Other packages list        | Todo                                        |
| ------------------------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------- |
| Basic code                           | [python](python/basic_code.ipynb)              | We introduce basic code used in this tutorial.                                               |                            |                                             |
| Characteristic                       | [python](python/characteristic.ipynb)          | Done (ja)                                                                                    |                            | translation to english                      |
| Nearest neighbors search             | [python](python/nns.ipynb)                     | kNN, Radius Search, Hybrid Search                                                            | [python, C++](docs/nns.md) |                                             |
| Tree structure                       | [python](python/tree_structure.ipynb)          | None                                                                                         |                            |                                             |
| Downsampling                         | [python](python/downsampling.ipynb)            | Random Sampling, Furthest point sampling, Voxel grid sampling                                |                            |                                             |
| Normal estimation                    | [python](python/normal_estimation.ipynb)       | Estimation with PCA, Normal re-orientation methods                                           |                            |                                             |
| Handcrafted feature                  | [python](python/handcrafted_feature.ipynb)     | FPH, PPF                                                                                     |                            |                                             |
| Deep learning                        | [python](python/deep_learning.ipynb)           | VoxNet, PointNet, PointNet++                                                                 |                            | add VoxNet impl. and translation to english |
| Visualization                        | [python](python/visualization.ipynb)           | Visualization functions in the repository                                                    |                            |                                             |
| Competition                          | [python](python/competition.ipynb)             | None                                                                                         |                            | add description                             |
| Transformation/Affine transformation | [python](python/affine_transformations.ipynb)  | Affine transformation, Transformation matrix                                                 |                            |                                             |
| Transformation/Camera projection     | [python](python/camera_projection.ipynb)       | Camera projection                                                                            |                            |                                             |
| Task/Reconstruction                  | [python](python/reconstruction.ipynb)          | Marching cube, meshes to points                                                              |                            | add description                             |
| Task/Registration                    | [python](python/registration.ipynb)            | ICP, RANSAC with Handcrafted features                                                        |                            | add description                             |
| Dataset/Pix3D                        | [python](python/datasets/pix3d.ipynb)          | [Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling](http://pix3d.csail.mit.edu/) |                            |                                             |
| Dataset/Redwood 3DScan               | [python](python/datasets/redwood_3dscan.ipynb) | [A Large Dataset of Object Scans](http://redwood-data.org/3dscan/)                           |                            |                                             |
| Dataset/Redwood Indoor               | [python](python/datasets/redwood_indoor.ipynb) | [Robust Reconstruction of Indoor Scenes](http://redwood-data.org/indoor/index.html)          |                            |                                             |
| Dataset/ScanNet                      | [python](python/datasets/scannet.ipynb)        | [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](http://www.scan-net.org/)    |                            |                                             |
| Dataset/Sun3D                        | [python](python/datasets/sun3d.ipynb)          | [SUN3D database](http://sun3d.cs.princeton.edu/)                                             |                            |                                             |

## About correction
If you finded any corrections, let us know in Issues.
