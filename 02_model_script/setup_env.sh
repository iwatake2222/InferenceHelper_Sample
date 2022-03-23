### On WSL2
docker create  -v /mnt/c/iwatake/devel:/root/devel -v /etc/localtime:/etc/localtime:ro -it --name=ubuntu20_model_generation ubuntu:20.04
docker start ubuntu20_model_generation
docker exec -it ubuntu20_model_generation bash


### On ubuntu20_model_generation (Client OS)
apt update
apt install -y python3 python3-pip wget ncurses-term protobuf-compiler

# For TensorFLow
pip3 install tensorflow==2.8 tensorflow_datasets

# For PyTorch
pip3 install torch==1.11.0 torchvision torchaudio

# For ONNX
pip3 install onnx-simplifier

# For TensorFlow Lite + EdgeTPU
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
apt update
apt install -y edgetpu-compiler

# For OpenVINO
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18319/l_openvino_toolkit_p_2021.4.752.tgz
tar xzvf l_openvino_toolkit_p_2021.4.752.tgz
cd l_openvino_toolkit_p_2021.4.752
./install_openvino_dependencies.sh
./install.sh
pip3 install -r /opt/intel/openvino_2021/deployment_tools/model_optimizer/requirements.txt
