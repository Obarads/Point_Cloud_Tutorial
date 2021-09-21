FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# install ubuntu package
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    gdb \
    wget \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libpng-dev \
    lxde \
    mesa-utils \
    ne \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    python3-dbg \
    pybind11-dev \
    software-properties-common \
    x11vnc \
    xorg-dev \
    xterm \
    ninja-build \
    xvfb

### PCL (Point Cloud Library)
RUN apt-get install -y libpcl-dev

## clean
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# install python packages
## conda
RUN conda update -n base -c defaults conda
RUN conda install cudatoolkit=11.1

### pytorch3D
RUN conda install -c fvcore -c iopath -c conda-forge fvcore=0.1.5.post20210804 iopath=0.1.9 -y
RUN conda install -c pytorch3d pytorch3d=0.5.0 -y

### toch-geometric
# RUN conda install -c rusty1s pytorch-geometric=1.7.2=py37_torch_1.9.0_cu111 -y
RUN conda install -c rusty1s -c conda-forge pytorch-geometric=1.7.2 -y

### open3d
RUN conda install -c open3d-admin  -c conda-forge open3d=0.13.0 -y

### other
RUN conda install -c conda-forge k3d=2.9.7 -y

### clren and init
RUN conda clean -i -t -y
RUN conda init

## pip
RUN pip install plyfile==0.7.2
RUN pip install opencv-python==4.5.3.56
