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
このsubsectionの内容は更新予定です。

### 1. 環境用意
動作環境については、`docker/Dockerfile`によるdockerイメージとコンテナ作成を行うか、`Dockerfile`に準ずる環境を用意してください。ただし、本チュートリアルは深層学習や一部コードを除いて、基本的にはCPUで動かすことを想定しています。現在`Dockerfile`ではGPU用の環境を指定していますが、後にCPU用の`Dockerfile`を用意する予定です。

環境用意後に、`git clone`で本レポジトリを環境内にダウンロードしてください。

```bash
git clone 
```

### 2. チュートリアルの開始
本レポジトリを動作環境にダウンロードした後、`python`内のipynbファイルを開きます。ipynbファイルは[VSCode](https://code.visualstudio.com/)や[jupyter](https://jupyter.org/)などを利用してファイルを開いてください。

