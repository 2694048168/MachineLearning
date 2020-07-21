---
title: 深度学习环境搭建详细教程
copyright: true
top: true
date: 2020-07-20 15:33:14
categories:
  - ML & DL
tags:
  - MachineLearning
---

# 开发环境 platform
- 操作系统 OS
    - Windows 7
    - Windows 10
    - Ubuntu 18.04
    - Ubuntu 16.04
    - Ubuntu 20.04
- 开发语言 Python3
    - Anaconda3
    - Jupyter Lab & Jupyter Notebook
    - PyCharm
- 开发框架
    - PyTorch
    - TensorFlow2
- 计算资源
    - CPU
    - GPU ( CUDA & cuDNN )
    - TPU

# 版本匹配 select
- Windows 7 + Anaconda3 + Jupyter Lab + Jupyter Notebook + PyCharm + PyTorch + TensorFlow + CPU + GPU + TPU + CUDA
- Windows 10 + Anaconda3 + Jupyter Lab + Jupyter Notebook + PyCharm + TensorFlow + CPU + GPU + TPU + CUDA + cnDNN
- Ubuntu 18.04 + Anaconda3 + Jupyter Lab + Jupyter Notebook + PyCharm + PyTorch + TensorFlow + CPU + GPU + TPU + CUDA
- Ubuntu 16.04 + Anaconda3 + Jupyter Lab + Jupyter Notebook + PyCharm + TensorFlow + CPU + GPU + TPU + CUDA + cnDNN

![self](./image/Platform.png)

# 实战顺序 order
1. Anaconda3
2. Jupyter
3. CUDA & cuDNN
4. PyTorch & TensorFlow
5. PyCharm

***software：安装路径；系统环境变量配置；快速启动与关闭；软件本身的一些相关配置***

# 安装命令 command
- Windows OS

```shell
# 1、安装 Anaconda3
# Anaconda 官网:https://www.anaconda.com/products/individual#windows
# Anaconda 国内清华大学镜像:https://mirrors.tuna.tsinghua.edu.cn/
# Windows 下【next】即可，注意选择将路径添加到系统环境变量中那个选项
# Win + R，打开运行窗口，输入 cmd
# 检测 Anaconda 安装情况
where conda    # 检测位置是否是自己想要的
where pip
where python
where ipython

conda --version #或者输入 conda -V  查看版本是否自己需要的
pip --version
python --version
ipython --version

# 升级 Anaconda 自带的 Jupyter
# 建议先配置好 pip 国内镜像源
pip install --upgrade jupyterlab
# cmd 打开 jupyter lab
jupyter lab
# 配置 jupyter ，一般设置工作路径，远程登录等等信息
# 生成配置文件
jupyter notebook --generate-config


# 2、NVIDIA 显卡资源 GTX 1060、GTX 1080Ti
# CUDA 10.0 官网：https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
# 右击以管理员身份运行，选择自定义安装，按需选择，注意与本身驱动版本区别匹配
# cuDNN 7.5 for CUDA 10.0 官网：https://developer.nvidia.com/rdp/cudnn-archive
# cuDNN 需要登录才能下载，注册并登录一些即可
# 解压后重命名文件夹为 cudnn ，然后剪切到 CUDA 安装目录下即可
######## 添加系统环境变量 PATH ############
# 1、CUDA\V10.0\bin
# 2、CUDA\V10.0\libnvvp
# 3、CUDA\V10.0\extras\CUPI\libx64
# 4、CUDA\V10.0\cudnn\bin
######## 四个环境变量一个不能少，而且必须保持到最前面 ############
# 测试
nvcc --version

# 3、配置好 pip 国内镜像源
# 查看包管理器pip的版本，至少要求pip版本在10以上
pip --version
# 如果有需要，可以升级pip到最新版本
python -m pip install --upgrade pip
# 设置全局默认pypi国内镜像源地址，只需要一个即可
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# 可以临时使用指定的pypi镜像源，命令如下
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple packageName
# 设置配置文件 Windows 
# Win + R 打开【运行】窗口，输入%HOMEPATH%，自动跳转到用户目录
# 在打开的目录下新建一个目录，命名为pip
# 进入pip目录，新建配置文件，命名为pip.ini，注意Windows下配置文件后缀为.ini
# 打开配置文件pip.ini，输入pypi源，如下内容
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com

# 4、安装 PyTorch
# PyTorch 官网：https://pytorch.org/get-started/locally/
# 选择对应的 PyTorch 版本，操作系统，包管理工具，支持语言，CUDA 版本
# 自动生成命令
# 例如：PyTorch 1.5.1，Windows 10，pip，Python，CUDA 10.2
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html

# 测试 PyTorch，进入 cmd 
# 没有 GPU 资源，不报错即可
ipython
import torch
# 有 GPU 资源，返回 True 即可
ipython
import torch
torch.cuda.is_available()

# 5、安装 TensorFlow2
# TensorFlow官网：https://www.tensorflow.org/install
# 安装 仅支持 CPU 版本
pip install tensorflow-cpu
# 测试,进入 cmd，输入命令没有错误即可
ipython
import tensorflow as tf

# 安装 支持 GPU和CPU 版本
pip install tensorflow
# 测试,进入 cmd，输入命令返回 True 即可
ipython
import tensorflow as tf
tf.test.is_gpu_available()

# 6、安装 PyCharm
# PyCharm 官网：https://www.jetbrains.com/pycharm/download/
# Windows 下【next】即可，注意选择将路径添加到系统环境变量中那个选项
# 注意安装时选择自定义安装，系统环境变量等等情况
# 快速启动并配置 Python 解释器等等一系列配置

```

- Linux-Ubuntu OS

```shell
# 1、配置国内 ubuntu 软件镜像源
sudo cp /etc/apt/sources.list /etc/apt/sources.list_backup
# 查看系统的版本号或者系统代号，便于配置相适应的镜像源
# Ubuntu 12.04 (LTS)代号为precise
# Ubuntu 14.04 (LTS)代号为trusty
# Ubuntu 16.04 (LTS)代号为xenial
# Ubuntu 18.04 (LTS)代号为bionic
lsb_release -c

# 2、安装 Anaconda3
# Anaconda 官网:https://www.anaconda.com/products/individual#windows
# Anaconda 国内清华大学镜像:https://mirrors.tuna.tsinghua.edu.cn/
# 添加执行权限
chmod +x anaconda3.sh
# 执行安装
./anaconda3.sh
# 1、回车开始安装
# 2、按 q 退出查看阅读协议
# 3、输入 yes 同意协议服务
# 4、输入安装路径 /home/user/conda/
# 5、添加系统环境变量 yes

# 检测 Anaconda 安装情况
which conda    # 检测位置是否是自己想要的
which pip
which python
which ipython

conda --version #或者输入 conda -V  查看版本是否自己需要的
pip --version
python --version
ipython --version

# 升级 Anaconda 自带的 Jupyter
# 建议先配置好 pip 国内镜像源
pip install --upgrade jupyterlab
# cmd 打开 jupyter lab
jupyter lab
# 配置 jupyter ，一般设置工作路径，远程登录等等信息
# 生成配置文件
jupyter notebook --generate-config

# 3、配置国内 pip 镜像源
# 在终端使用如下命令，新建 pip 配置文件，为当前登录用户 Python 设置 pypi 镜像源
vi ~/.pip/pip.config
# 在配置文件中输入 pypi 源，需要简单的 Vi&Vim 操作命令
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com

# 4、安装 GPU 驱动加速以及 cuDNN 深度神经网络加速库
# 显卡信息查看
lspci
# 安装 CUDA，搜索 cuda 10.2 download 即可
# 选择好 cuda版本、操作系统、架构64、ubuntu以及其版本、deb local
# 出现安装指南 Installation Instructions:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# 重启使得网卡驱动生效
# 查看信息,显示 GPU 资源信息即可
nvidia-smi
# 配置 nvcc 到系统环境变量，并使其生效
# 查看当前 PATH 内容
echo $PATH
# 找到 cuda 安装路径下的 nvcc 命令
# 类似：/usr/local/cuda_10.2/bin
vi ~/.bashrc
# 添加内容
export PATH="/usr/local/cuda_10.2/bin:$PATH"
# 生效配置文件
source ~/.bashrc
# 检测 nvcc
nvcc -V

# 5、安装 PyTorch
# PyTorch 官网：https://pytorch.org/get-started/locally/
# 选择对应的 PyTorch 版本，操作系统，包管理工具，支持语言，CUDA 版本
# 自动生成命令
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# 测试 PyTorch，打开终端 Ctrl + Alt + T
# 没有 GPU 资源，不报错即可
ipython
import torch
# 有 GPU 资源，返回 True 即可
ipython
import torch
torch.cuda.is_available()

# 6、安装 TensorFlow2
# TensorFlow官网：https://www.tensorflow.org/install
# 安装 仅支持 CPU 版本
pip install tensorflow-cpu
# 测试,进入 cmd，输入命令没有错误即可
ipython
import tensorflow as tf

# 安装 支持 GPU和CPU 版本
pip install tensorflow
# 测试,进入 cmd，输入命令返回 True 即可
ipython
import tensorflow as tf
tf.test.is_gpu_available()

# 7、注意点
# 之所有没有安装 cuDNN 库支持，并能成功使用 TensorFlow  的 GPU 资源
# 这是因为，PyTorch 安装中自动下载并配置了 cudatoolkit 库，即就是 cuDNN 库
# 如果没有安装 PyTorch，则需要自己配置 cuDNN 库
# cuDNN 7.5 for CUDA 10.0 官网：https://developer.nvidia.com/rdp/cudnn-archive
# cuDNN 需要登录才能下载，注册并登录一些即可
# 下载并解压后重命名文件夹为 cudnn ，然后 mv 到 CUDA 安装目录下即可
# 配置 cudnn 库到系统环境变量库中,指定 cudnn 库位置
echo $LD_LIBRARY_PATH
vi ~/.bashrc
# 添加内容
export LD_LIBRARY_PATH="/home/cuda/cudnn/lib64:$LD_LIBRARY_PATH"
# 生效配置文件
source ~/.bashrc

# 7、安装 PyCharm
# PyCharm 官网：https://www.jetbrains.com/pycharm/download/
# 注意安装时选择自定义安装，系统环境变量等等情况
# 解压，执行 pycharm.sh
./pycharm.sh
# 按照提示进行安装即可
# 选择快速启动命令 charm 终端启动

```

# 基本概念 conception

## 系统环境变量
- 功能：使得在系统终端能够搜索并使用某个命令或者搜索到某个动静态库

## 编译器
- 功能：将源代码编译成机器指令，供计算机执行

## 编辑器
- 功能：快速编辑源代码

## 调试器
- 功能：对程序进行调试，找出错误bug

## 集成开发环境 IDE
- 编辑 + 编译 + 调试 + Git版本控制

## 操作系统
### Windows 系列
- Windows XP
- Windows 7
- Windows 8
- Windows 10

### Linux 系列
- Unix
- Ubuntu
- Debian
- Red Hat
- Centos
- Arch Linux
- Linux Mint
- Android

### Mac 系列
- Mac OS 9
- Mac OS X 10.0
- Mac OS X 10.12
- Mac OS X 10.14
- Mac OS X 10.15
- iOS


## 深度学习框架
- Scikit-learn    for Machine Learning , no GPU
- Keras    a Deep Learning library
- 2013年，第一个面向深度学习的框架 Caffe ，C-plus-plus 开发，不支持自动求导
- Theano，加拿大，开发难，调试难
- Torch，采用 Lua 语言
- TensorFlow，Google，2017发布TensorFlow1，2019发布TensorFlow2
- PyTorch，Facebook
- Chainer ，日本
- MXNet

**现在深度学习框架**
- TensorFlow + Keras( Keras 作为 TensorFlow 后端)
- PyTorch + Caffe2（Caffe2 作为 PyTorch 后端，Torch 作为 PyTorch 前端）
- 优势：GPU加速，自动求导，神经网络API


## 人工智能常用库
- Scikit-Learn 机器学习库，分类、回归、数据处理、降维处理、聚类、模型选择
[scikit-learn官网](https://scikit-learn.org/stable/)
[scikit-learn-GitHub]( https://github.com/scikit-learn/scikit-learn/)
[scikit-learn-Gitee](https://gitee.com/mirrors/scikit-learn/)
- OpenCV 图像处理库，图像视频处理
[OpenCV官网](https://opencv.org/)
[OpenCV-GitHub](https://github.com/opencv/opencv)
[OpenCV-Gitee](https://gitee.com/mirrors/opencv/)
- PyTorch
[PyTorch官网](https://pytorch.org/)
[PyTorch-GitHub](https://github.com/pytorch/pytorch)
[PyTorch-Gitee](https://gitee.com/mirrors/pytorch/)
- TensorFlow
[TensorFlow官网](https://www.tensorflow.org/)
[TensorFlow-GitHub](https://github.com/tensorflow/tensorflow)
[TensorFlow-Gitee](https://gitee.com/mirrors/tensorflow/)
- Scipy 科学计算基础库
[Scipy官网](https://www.scipy.org/)
[Scipy-GitHub]( https://github.com/donnemartin/data-science-ipython-notebooks/)
[Scipy-Gitee](https://gitee.com/mirrors/data-science-ipython-notebooks/)
-  Numpy 矩阵高效运算
[Numpy官网](https://numpy.org/)
[Numpy-GitHub](https://github.com/numpy/numpy/)
[Numpy-Gitee](https://gitee.com/mirrors/NumPy/)
- Pandas 数据快速处理
[Pandas官网](https://pandas.pydata.org/)
[Pandas-GitHub]( https://github.com/pandas-dev/pandas/)
[Pandas-Gitee](https://gitee.com/mirrors/pandas/)
- Matplotlib 可视化绘制
[Matplotlib官网](https://matplotlib.org/)
[Matplotlib-GitHub](https://github.com/matplotlib/matplotlib/)
[Matplotlib-Gitee](https://gitee.com/mirrors/matplotlib/)
- Git 代码版本管理工具
[Git管理工具官网](https://git-scm.com/)
[Git-GitHub](https://github.com/git/git/)
[Git-Gitee](https://gitee.com/mirrors/git/)


## 国内镜像源
### Linux OS
- [阿里云镜像](https://developer.aliyun.com/mirror/)
- [华为镜像](https://mirrors.huaweicloud.com/os/image/)
- [中科院](http://mirrors.ustc.edu.cn/)
- [清华大学](https://mirrors.tuna.tsinghua.edu.cn/)


### Python包 PyPi
- [阿里云镜像](https://developer.aliyun.com/mirror/pypi/)
- [清华大学](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)


### Ubuntu 软件源
- [阿里云镜像](https://developer.aliyun.com/mirror/ubuntu/)
- [清华大学](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)


# 简要说明 introduction
- 清晰思路，整个安装过程一定要思路清晰，每一步都需要知道在干啥子
- 本文仅做参考，可能随着岁间流逝，有些命令有所变换，笔者尽量做到最新，希望更确定命令参考官方说明文档
- 笔者研究方向：数字图像处理 DIP、计算机视觉 CV
- 笔记信条：尽量操作过程不要展示图片，入这门，思维抽象很重要，这些操作必是熟稔于心
- 欢迎对内容进行补充和纠错，能够帮助更多的人！
- 评论或者Email@ 
- Email：2694048168@qq.com
