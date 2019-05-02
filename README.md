# [Run Keras/Tensorflow model on RK3399Pro](https://www.dlology.com/blog/how-to-run-keras-model-on-rk3399pro/)

### Clone or download this repo
```
git clone https://github.com/Tony607/Keras_RK3399pro
```

**Download pre-compiled Python wheel files from my [aarch64_python_packages](https://coding.net/u/zcw607/p/aarch64_python_packages/git) repo and [rknn_toolkit](https://github.com/rockchip-toybrick/RKNPUTool/tree/master/rknn-toolkit/package) wheels from their official GitHub.** 
### Step1: Freeze Keras model and convert to RKNN model (On Linux development machine)
Require [Python 3.5+](https://www.python.org/ftp/python/3.6.7/python-3.6.7.exe).

### Install required libraries for your development machine
`pip3 install -r requirements.txt`

The install rknn toolkit with the following command.
```
pip3 install rknn_toolkit-0.9.9-cp36-cp36m-linux_x86_64.whl
```

To freeze a Keras InceptionV3 ImageNet model to a single `.pb` file.
The frozen graph will accept inputs with shape `(N, 299, 299, 3)`.
```
freeze_graph.py
```

To convert the `.pb` file to `.rknn` file, run
```
python3 convert_rknn.py
```

### Step2: Make prediction (On RK3399Pro board)
Setup for the first time.
```bash
sudo dnf update -y
sudo dnf install -y cmake gcc gcc-c++ protobuf-devel protobuf-compiler lapack-devel
sudo dnf install -y python3-devel python3-opencv python3-numpy-f2py python3-h5py python3-lmdb
sudo dnf install -y python3-grpcio

sudo pip3 install scipy-1.2.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install onnx-1.4.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install tensorflow-1.10.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install rknn_toolkit-0.9.9-cp36-cp36m-linux_aarch64.whl
```


To run inference benchmark on RK3399Pro board, in its terminal run,
```
python3 benchmark_incption_v3.py
```