# Point cloud tutorial
This repository contains tutorial code and supplementary note for point cloud processing. In order to understand algorithm of point cloud processing, most codes are implemented with [Numpy](https://numpy.org/) and [Jupyter](https://jupyter.org/).

## Repository structure
```bash
┌─ data             # 3D files for examples
├─ .devcontainer    # Dockerfile for this tutorial
└─ python           # tutrial codes
    ├─ tutlibs      # package for tutorial codes
    └─ *.ipynb      # tutorial codes
```

## How to use
### 1. Enviroment
You can execute most tutorial codes on [Codespaces](https://github.com/features/codespaces)(CPU resource). If you have GPU resource, can execute all tutorial codes. For more environment of CPU and GPU, Please refer to [.devcontainer/README.md](.devcontainer/README.md).

### 2. About tutorial
Tutorial contents are as follows:

| Theme               | Page & code                                | Abstract                                                      | Todo                                        |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------- |
| basic code          | [python](python/basic_code.ipynb)          | We introduce basic code used in this tutorial.                |                                             |
| characteristic      | [python](python/characteristic.ipynb)      | Done (ja)                                                     | translation to english                      |
| competition         | [python](python/competition.ipynb)         | To do                                                         |                                             |
| handcrafted feature | [python](python/handcrafted_feature.ipynb) | Done (ja)                                                     | translation to english                      |
| deep learning       | [python](python/deep_learning.ipynb)       | To do                                                         |                                             |
| nns                 | [python](python/nns.ipynb)                 | Nearest neighbors search                                      |                                             |
| normal estimation   | [python](python/normal_estimation.ipynb)   | Normal estimation with a point cloud                          |                                             |
| registration        | [python](python/registration.ipynb)        | To do                                                         | add description                             |
| downsampling        | [python](python/downsampling.ipynb)        | Downsampling for point cloud processing                       |                                             |
| transformations     | [python](python/transformations.ipynb)     | Transfomation and Transfomation matrix                        | add 2D-3D transfomation (from constructing) |
| visualization       | [python](python/visualization.ipynb)       | Visualization functions in the repository                     |                                             |
| constructing        | [python](python/constructing.ipynb)        | Constructing 3D/2.5D representation from other representation | add description                             |



## About correction
If you finded any corrections, let us know in Issues.
