# Run Keras/Tensorflow model on RK3399Pro

### Clone or download this repo
```
git clone https://github.com/Tony607/Keras_RK3399pro
```


### Step1: Freeze Keras model and convert to RKNN model (On Linux development machine)
Require [Python 3.5+](https://www.python.org/ftp/python/3.6.7/python-3.6.7.exe) and [Jupyter notebook](https://jupyter.readthedocs.io/en/latest/install.html) installed

### Install required libraries for your development machine
`pip3 install -r requirements.txt`

The install rknn toolkit with the following command.
```
pip3 install rknn_toolkit-0.9.9-cp36-cp36m-linux_x86_64.whl
```
Start a terminal on your development machine, then run,
```
jupyter notebook
```

In the opened browser window open
```
freeze_graph.ipynb
```

To convert the `.pb` file to `.rknn` file, run
```
python3 convert_rknn.py
```

### Step2: Make prediction (On RK3399Pro board)
Setup for the first time.
```bash
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