# docker
## How to make a GPU or CPU env. with docker build.
- GPU
    ```bash
    # build and run
    wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile 
    docker build . -t pct-gpu
    docker run -dit -p 8888:8888 --gpus all --name pct_gpu pct-gpu
    # execute Jupyter
    docker exec pct_gpu git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /root/workspace/Point_Cloud_Tutorial
    ```
- CPU
    ```bash
    # build and run
    wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile_cpu
    docker build . -t pct-cpu -f Dockerfile_cpu
    docker run -dit -p 8888:8888 --name pct_cpu pct-cpu
    # execute Jupyter
    docker exec pct_cpu git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /root/workspace/Point_Cloud_Tutorial
    ```


