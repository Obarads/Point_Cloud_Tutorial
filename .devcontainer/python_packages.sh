set -eu

if [ "$#" -ne 1 ]
then
    echo "Usage: `./pip.sh gpu` or `./pip.sh cpu`"
    exit 1
fi

pip3 install open3d==0.13.0 k3d==2.11.0 plyfile==0.7.2 pandas==1.3.4 h5py==3.6.0
pip3 install opencv-python

echo $1

if [ $1 = "gpu" ]; then
    echo "with GPU"
    pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    conda install --name base jupyterlab --update-deps --force-reinstall
else
    echo "withiout GPU"
    pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi
