# nvidia/cuda:11.4.3-devel-ubuntu20.04
apt update && apt-get install -y sudo vim
sudo apt upgrade -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.8 python3.8-venv
sudo update-alternatives --install -y /usr/bin/python python /usr/bin/python3.8 1
sudo ln -s /usr/bin/python3.8 /usr/bin/python
sudo apt install -y python3-pip