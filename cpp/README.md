# CPP
## PCLについて
- PCLのインストールは以下で終わり
    ```
    apt-get -y install libpcl-dev
    ```
- ファイルやその他ファイルの位置関係は以下の通り。
    ```bash
    ┌─ build            # build folder
    ├─ CMakeLists.txt
    └─ main.cpp
    ```

- build時には`CMakeLists.txt`ファイルと`build`ディレクトリが必要になる。
  - 使いたい機能に合わせて、CMakeLists.txt内の`find_package`のパッケージ名(kdtree search fearuresなどの空白区切りの表記)を追加する。
- buildコマンドは以下の通り。
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```