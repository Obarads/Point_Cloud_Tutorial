set -eu

if [ "$#" -ne 1 ]
then
    echo "Usage: `./pip.sh gpu` or `./pip.sh cpu`"
    exit 1
fi

pip3 install open3d==0.13.0 k3d==2.11.0 plyfile==0.7.2 pandas==1.3.4
pip3 install opencv-python

if [ $1 == "gpu" ]; then
    echo "with GPU"
    pip3 install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "withiout GPU"
    pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
fi
