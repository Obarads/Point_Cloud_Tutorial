# Point cloud tutorial
ipynbファイルを使った点群処理のpythonチュートリアルレポジトリです。

## Repository structure
```bash
┌─ cpp              # C++ codes for tests
├─ data             # 3D files for examples
├─ .devcontainer    # Dockerfile for this tutorial
└─ python           # tutrial codes
    ├─ tutlibs      # package for tutorial codes
    └─ *.ipynb      # tutorial codes
```

## How to use
### 1. 環境用意
動作環境については、`.devcontainer/Dockerfile`によるdockerイメージとコンテナ作成を行うか、`Dockerfile`に準ずる環境を用意してください。作成後、環境内に本レポジトリをクローンしてください。

本チュートリアルは最低限の実装を載せているため、深層学習や一部コードを除いて、基本的にはCPUで動かすことを想定しています。また、動作環境を統一するために以下の環境を作成予定または用意しています。
- Docker
  - Dockerfileによるコンテナ作成 (以下に載せる例はこの方法を使用)
  - .devcontainerによるVSCodeからのコンテナ作成
- Codespaces
  - GithubからCodespaces上での作成
    - Codespaces環境のJupyterのKernelは`Python 3.8.X 64-bit`を利用、視覚化時に`Widgets require us ...`とパネルが出るので、`Ok`を押す。

GPUとdockerを用いた環境の用意を行う場合は、以下のコマンドを入力します。
1. Dockerによるコンテナの作成
    ```bash
    # Get Dockerfile from Github.
    wget https://raw.githubusercontent.com/Obarads/Point_Cloud_Tutorial/main/.devcontainer/Dockerfile 
    # Create Image.
    docker build . -t pct-gpu
    # Create a container with a port (-p). If you do not use GPU, please remove `--gpus all`.
    docker run -dit -p 8888:8888 --gpus all --name pctut pct 
    ```
2. 本レポジトリを環境にダウンロードします。
    ```bash
    docker exec pctut git clone https://github.com/Obarads/Point_Cloud_Tutorial.git /root/workspace/Point_Cloud_Tutorial
    ```

### 2. チュートリアルのファイル表示
チュートリアルのファイルは`python`フォルダに入っており、ファイルでは`ipynb`を使用しています。ipynbファイルは[VSCode](https://code.visualstudio.com/)や[jupyter](https://jupyter.org/)などを介して表示できます。各チュートリアルの概要については、pythonフォルダ内の`README.md`をご覧ください。

上のsubsectionの例を実行している場合、jupyterの起動は以下のコマンドで可能です。
1. Jupyter notebookの起動
    ```bash
    docker exec pctut jupyter notebook --notebook-dir /root/workspace/Point_Cloud_Tutorial --allow-root --port 8888 --ip=0.0.0.0
   ```
2. コマンドを入力した後、コマンドを入力した端末にjupyterのアクセス先( http://127.0.1:8888/..... )が出るため、そのURLへアクセスする。

## About correction
もし修正点がある場合は、Issuesでお知らせください。
