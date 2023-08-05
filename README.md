# Point cloud tutorial

[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/) 
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

<!-- [![Packaged with Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/) -->

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

**Basic**

| Theme                                | Page & code                                   | Contents                                                      | Other packages list        | Todo            |
| ------------------------------------ | --------------------------------------------- | ------------------------------------------------------------- | -------------------------- | --------------- |
| Basic code                           | [python](python/basic_code.ipynb)             | We introduce basic code used in this tutorial.                |                            |                 |
| Characteristic                       | [python](python/characteristic.ipynb)         | characteristic of the point cloud                             |                            |                 |
| Nearest neighbors search             | [python](python/nns.ipynb)                    | kNN, Radius Search, Hybrid Search                             | [python, C++](docs/nns.md) |                 |
| Tree structure                       | [python](python/tree_structure.ipynb)         | None                                                          |                            |                 |
| Downsampling                         | [python](python/downsampling.ipynb)           | Random Sampling, Furthest point sampling, Voxel grid sampling |                            |                 |
| Normal estimation                    | [python](python/normal_estimation.ipynb)      | Estimation with PCA, Normal re-orientation methods            |                            |                 |
| Handcrafted feature                  | [python](python/handcrafted_feature.ipynb)    | FPH, PPF                                                      |                            |                 |
| Visualization                        | [python](python/visualization.ipynb)          | Visualization functions in the repository                     |                            |                 |
| Competition                          | [python](python/competition.ipynb)            | None                                                          |                            | add description |
| Transformation/Affine transformation | [python](python/affine_transformations.ipynb) | Affine transformation, Transformation matrix                  |                            |                 |
| Transformation/Camera projection     | [python](python/camera_projection.ipynb)      | Camera projection                                             |                            |                 |
| Task/Reconstruction                  | [python](python/reconstruction.ipynb)         | Marching cube, meshes to points                               |                            | add description |
| Task/Registration                    | [python](python/registration.ipynb)           | ICP, RANSAC with Handcrafted features                         |                            | add description |
| Task/SLAM                            | [python](ptyhon/slam.ipynb)                   | None                                                          | [python](docs/slam.md)     |                 |

**Deep Learning**
| Theme      | Page & code                                        | Paper name                                                                                                                                                | Other packages list | Todo                   |
| ---------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ---------------------- |
| PointNet   | [python](python/deep_learning/pointnet.ipynb)      | [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)                                          |                     | translation to english |
| PointNet++ | [python](python/deep_learning/pointnet2.ipynb)     | [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)                                        |                     | translation to english |
| VoxNet     | [python](python/deep_learning/pointnet.ipynb)      | [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recongnition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf) |                     | translation to english |
| ASIS       | [python](python/deep_learning/pointnet2asis.ipynb) | [Associatively Segmenting Instances and Semantics in Point Clouds](https://arxiv.org/abs/1902.09852)                                                      |                     | translation to english |

**Dataset**
| Dataset Name             | Page & code                                    | Paper name                                                                                   | Other packages list | Todo |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------- | ---- |
| A2D2 (with Bounding box) | [python](python/datasets/a2d2.ipynb)           | [A2D2: Audi Autonomous Driving Dataset](https://www.a2d2.audi/a2d2/en.html)                  |                     |      |
| Pix3D                    | [python](python/datasets/pix3d.ipynb)          | [Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling](http://pix3d.csail.mit.edu/) |                     |      |
| Redwood 3DScan           | [python](python/datasets/redwood_3dscan.ipynb) | [A Large Dataset of Object Scans](http://redwood-data.org/3dscan/)                           |                     |      |
| Redwood Indoor           | [python](python/datasets/redwood_indoor.ipynb) | [Robust Reconstruction of Indoor Scenes](http://redwood-data.org/indoor/index.html)          |                     |      |
| ScanNet                  | [python](python/datasets/scannet.ipynb)        | [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](http://www.scan-net.org/)    |                     |      |
| Sun3D                    | [python](python/datasets/sun3d.ipynb)          | [SUN3D database](http://sun3d.cs.princeton.edu/)                                             |                     |      |

**Experiments**
A place to write the results of the experiments I've done.

| Experiments Name                           | Page & code | References |
| ------------------------------------------ | ----------- | ---------- |
| SuperPointGraph with instance segmentation |             |            |

## About correction
If you finded any corrections, let us know in Issues.
