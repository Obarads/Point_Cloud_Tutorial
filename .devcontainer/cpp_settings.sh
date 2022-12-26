# Install cmake
wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1.tar.gz
tar -zxvf cmake-3.25.1.tar.gz
rm cmake-3.25.1.tar.gz
cd cmake-3.25.1
./bootstrap
sudo make install
cd ..
rm -rf cmake-3.25.1/

# Download and set eigen
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout 311cc0f9cc66fa49523bbcb45a9ba22363fdd65a
sudo mv /usr/include/eigen3/Eigen /usr/include/eigen3/Eigen_prev
sudo mv Eigen/ /usr/include/eigen3/
cd ..
rm -rf eigen

# Download and set nanoflann
NANOFLANN_PATH=/usr/include/nanoflann
sudo mkdir $NANOFLANN_PATH
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
git checkout e105aacbdbf4b5738fac9ebf50b4fca73339606b
sudo mv include/nanoflann.hpp $NANOFLANN_PATH
sudo mv examples/utils.h $NANOFLANN_PATH
cd ../
rm -rf nanoflann
