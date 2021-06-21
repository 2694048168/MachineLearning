# Paper

TensorFlow Note 使用笔记

---------------------------------

## Files & Setting Description
```
 .
 ├── 00_test_tf_enviroment.py
 ├── ......
 ├── 31_gru_stock.py
 ├── checkpoint
 │   ├── cifar_AlexNet/
 │   ├── LSTM_stock/
 │   └── ......
 ├── csv_file
 │   ├── moutai.csv
 │   ├── GRU_stock.txt
 │   ├── ......
 ├── images
 │   ├── epoc0.png
 │   ├── iris_loss.png
 │   ├── iris_accuracy_SGDM.png
 │   ├── ......
 ├── logs
 │   ├── log_Adam.txt
 │   ├── ......
 └── README.md
```
```
- 00_test_tf_enviroment.py 测试 TensorFlow 环境
- checkpoint/ 训练过程中保存的权重文件，避免训练中断，可以重新接着训练
- images/ 所有保存或者测试需要的图像
- csv_file/ 所有需要的或者生成的文本文件
- logs/ 所有训练时候产生的日志文件
- README.md 项目说明文件(该本身文件)
```

-----------------------------
## Training
```python
python 31_gru_stock.py > log_train.txt
```

--------------------
## Testing
```python
# 可以选择使用 conda 包管理器进行虚拟环境的隔离
# python 3.8
# TensorFlow 2.5 以及对应的 CUDA 和 cuNDD 版本
# https://tensorflow.google.cn/install/source_windows

# 创建虚拟环境 python 隔离环境
python -m venv VENV_TF
# cd ~/VENV_TF/Scripts
# activate

# 建议使用国内镜像源镜像安装下载：https://blog.csdn.net/weixin_46782218/article/details/105311458
```