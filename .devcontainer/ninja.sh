wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
rm ninja-linux.zip
