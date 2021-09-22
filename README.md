# Point cloud tutorial
ipynbファイルを使った点群処理のpythonチュートリアルレポジトリです。

## Repository structure
```bash
┌─ cpp              # C++ codes for tests
├─ data             # 3D files for examples
├─ docker           # Dockerfile for this tutorial
└─ python           # tutrial codes
    ├─ tutlibs      # package for tutorial codes
    └─ *.ipynb      # tutorial codes
```

## How to use
### 1. 環境用意
動作環境については、`docker/Dockerfile`によるdockerイメージとコンテナ作成を行うか、`Dockerfile`に準ずる環境を用意してください。ただし、本チュートリアルは深層学習や一部コードを除いて、基本的にはCPUで動かすことを想定しています。作成後、環境内に本レポジトリをクローンしてください。

GPUとdockerを用いた環境の用意を行う場合は、以下のコマンドを入力します。
1. Dockerによるコンテナの作成
    ```bash
    # Get Dockerfile from Github.
    wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/docker/Dockerfile 
    # Create Image.
    docker build . -t pct
    # Create a container with a port (-p). If you do not use GPU, please remove `--gpus all`.
    docker run -dit -p 8888:8888 --gpus all --name pctut pct 
    ```
2. 本レポジトリを環境にダウンロードします。
    ```bash
    docker exec pctut git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /root/workspace/Point_Cloud_Tutorial
    ```

### 2. チュートリアルのファイル表示
チュートリアルのファイルは`python`フォルダに入っており、ファイルでは`ipynb`を使用しています。ipynbファイルは[VSCode](https://code.visualstudio.com/)や[jupyter](https://jupyter.org/)などを介して表示できます。各チュートリアルの概要については、pythonフォルダ内の`README.md`をご覧ください。

jupyterを介して表示を行う場合は、以下のコマンドを入力します。
1. Jupyter notebookの起動
    ```bash
    docker exec pctut jupyter notebook --notebook-dir /root/workspace/Point_Cloud_Tutorial --allow-root --port 8888 --ip=0.0.0.0
   ```
2. コマンドを入力した後、コマンドを入力した端末にjupyterのアクセス先(http://127.0.1:8888/.....)が出るため、そのURLへアクセスする。

