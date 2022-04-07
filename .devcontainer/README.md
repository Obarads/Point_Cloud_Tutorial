# docker
## How to make a CPU or GPU enviroment with docker build.
### CPU
You can create a GPU enviroment with devcontainer.json (via VSCode) or docker commands.
- devcontainer.json via VSCode:  
    Please check [devcontainer.json via VSCode](https://code.visualstudio.com/docs/remote/create-dev-container#_create-a-devcontainerjson-file). Since I have already prepared the devcontainer.json, all you have to do is run `Remote-Containers: Reopen in Container`.
- docker commands:
    ```bash
    # build an enviroment
    wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile.cpu
    docker build . -t pct-cpu -f Dockerfile.cpu
    docker run -dit -p 8888:8888 --name pct_cpu pct-cpu
    docker exec pct_cpu git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /workspace/Point_Cloud_Tutorial
    docker exec pct_cpu sh /workspaces/Point_Cloud_Tutorial/.devcontainer/python_packages.sh cpu
    ```

### GPU
You can create a GPU enviroment with docker commands.
- docker commands:
    ```bash
    # build an enviroment
    wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile.gpu
    docker build . -t pct-gpu -f Dockerfile.gpu
    docker run -dit -p 8888:8888 --gpus all --name pct_gpu pct-gpu
    docker exec pct_gpu git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /workspace/Point_Cloud_Tutorial
    docker exec pct_gpu sh /workspace/Point_Cloud_Tutorial/.devcontainer/python_packages.sh gpu

    # Optional settings for Codespaces bash prompt theme
    docker exec pct_gpu sh /workspace/Point_Cloud_Tutorial/.devcontainer/optional_setting.sh
    ```

## About tutorial directory and files
This tutorial folder will be saved into `/workspaces/Point_Cloud_Tutorial`. This tutorial use `ipynb` file that can be opened by [VSCode](https://code.visualstudio.com/) and [jupyter](https://jupyter.org/), etc.
For example, the steps to use jupyter are as follows:
1. Start jupyter notebook server.
    ```bash
    CONTAINER_NAME=<pct_gpu or pct_cpu>
    docker exec $CONTAINER_NAME jupyter notebook --notebook-dir /workspace/Point_Cloud_Tutorial --allow-root --port 8888 --ip=0.0.0.0
   ```
2. Access the URL (ex. http://127.0.1:8888/..... ) displayed on terminal.