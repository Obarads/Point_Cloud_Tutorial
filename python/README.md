# Point Cloud Tutorial
## 概要
点群をpythonで扱うためのチュートリアルです。このチュートリアルは、特徴抽出手法や特徴をタスクの結果に反映する方法を軸にしていることに注意してください。

## 環境
[Dockerfile](../docker/Dockerfile)で作成されたコンテナに従って本チュートリアルを進めます。Dockerfile内で書かれている内容については、[dockerのREADME.md](../docker/README.md)と以下のを参照してください。
点群処理においてはGPUを利用することでより効率的かつ短時間の処理を実現することが可能ですが、ここでは深層学習などの一部処理以外はCPUで動くように設計しています。これは、本チュートリアルが簡易な確認なども行える様に作成しているためです。

## 各ファイルの進捗
`ファイル名: 記載予定`という形式で以下に記載する。`記載予定`が「なし」のものは現状完了しているものです。
- basic_code.ipynb              : なし
- characteristic.ipynb          : なし
- converting.ipynb              : Mesh to pointとpoint to Meshのコードと説明を書く。
- coodinate_system.ipynb        : グローバルとローカル座標系について書く。
- features.ipynb                : handcrafted featureとdeep learningの例を書く。
- handcrafted_feature.ipynb     : PPFとPFHの計算があっているか確認する。
- nns.ipynb                     : kNNとrNNの説明を書く。(kNNは後少し)
- normals.ipynb                 : Normal estimationの自作を修正する、あと説明書く。
- sampling.ipynb                : コードの説明を書く。
- transformations.ipynb:        : なし

## チュートリアルで使用するパッケージ一覧
### Open3D
- [Code](https://github.com/isl-org/Open3D) | [Docs](http://www.open3d.org/docs/release/) | [ProjectPage](http://www.open3d.org/)
- 点群の操作(剛体変換やサンプリング)、ファイルIO、タスクに対する手法などを盛り込んだライブラリです。python版のPCLと見ることもできます。

### PyTorch geometric
- [Code](https://github.com/rusty1s/pytorch_geometric) | [Docs](https://pytorch-geometric.readthedocs.io/en/latest/)
- グラフやその他(不規則な構造)の表現をPytorchで扱うために生み出されたライブラリです。
- 個人Note: 上記の表現を扱った深層学習を[Geometric Deep Learning](https://geometricdeeplearning.com/)として呼んでいるらしい。

### Pytorch3D
- [Code](https://github.com/facebookresearch/pytorch3d) | [Docs](https://pytorch3d.readthedocs.io/en/latest/) | [ProjectPage](https://pytorch3d.org/)
- **概要**: Pytorchによる3D CV研究開発を効率的に行うために生み出されたライブラリです。
- **注意**: 本チュートリアルではまだ予定しているだけです。

### plyfile
- [Code](https://github.com/dranjan/python-plyfile)
- **概要**: PLYファイルの読み込み or 保存に使用するパッケージです。

### K3D Jupyter
- [Code](https://github.com/K3D-tools/K3D-jupyter) | [Docs](https://k3d-jupyter.org/)
- **概要**: 3D表現をJupyter上に視覚化することができるパッケージです。


## 使用しないが関連するパッケージ一覧
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
  - [MMDetection](https://github.com/open-mmlab/mmdetection)の開発元が作成している3D物体検出ライブラリです。最新(2021年現在)の手法もおいてあり、手法が充実しています。
  - 個人Note: ただし、ネットワークのbackboneが貧弱のようにも見える。
- [PyTorch Points 3D](https://github.com/nicolas-chaulet/torch-points3d)
- [python-pcl](https://github.com/strawlab/python-pcl)
- [Open3D-ML](https://github.com/isl-org/Open3D-ML)
- [pptk](https://github.com/heremaps/pptk)
- [vedo](https://github.com/marcomusy/vedo)
