# docker
## How to make a GPU or CPU enviroment with docker build.
### GPU
Please excute commands:
```bash
# build an enviroment
wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile 
docker build . -t pct-gpu -f Dockerfile_gpu
docker run -dit -p 8888:8888 --gpus all --name pct_gpu pct-gpu
docker exec pct_gpu git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /root/workspace/Point_Cloud_Tutorial
```

### CPU
Please excute commands:
```bash
# build an enviroment
wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile_cs
docker build . -t pct-cpu -f Dockerfile_cs
docker run -dit -p 8888:8888 --name pct_cpu pct-cpu
docker exec pct_cpu git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /root/workspace/Point_Cloud_Tutorial
docker exec pct_cpu sh /workspaces/Point_Cloud_Tutorial/.devcontainer/pip_cpu.sh
```

## About tutorial directory and files.
This tutorial folder will be saved into `/workspaces/Point_Cloud_Tutorial`. This tutorial use `ipynb` file that can be opened by [VSCode](https://code.visualstudio.com/) and [jupyter](https://jupyter.org/), etc.
For example, the steps to use jupyter are as follows:
1. Start jupyter notebook server.
    ```bash
    docker exec pctut jupyter notebook --notebook-dir /root/workspace/Point_Cloud_Tutorial --allow-root --port 8888 --ip=0.0.0.0
   ```
2. Access the URL (ex. http://127.0.1:8888/..... ) displayed on terminal.