#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 17_fashion_test.py
@Function: TensroFlow2 API tf.keras 实现模型的直接使用和测试
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage:    $ python 17_fashion_test.py 
# -----------------------------------
# -----------------------------------
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

tf.config.set_soft_device_placement(True)


# ------------------------------------------------
# 搭建神经网络的八股文
# TensorFlow2 API: tf.keras 搭建网络八股 
# ------------------------------------------------
# 1. import
# 2. train and test dataset
# 3. model = tf.keras.models
# 4. model.compile(optimizer=, loss=, metrics=[])
# 5. model.fit(train_data, train_label, batch_size=, epochs=, ...)
# 6. model.summary()
# ------------------------------------------------
# ------------------------------------------------
# 神经网络八股功能扩展
# TensorFlow2 API: tf.keras 扩展功能 
# ------------------------------------------------
# 1. 自己制作数据集，解决本领域应用
# 2. 数据增强，扩充数据集
# 3. 断点继续训练，存取模型
# 4. 参数提取，将参数存入文本
# 5. acc/loss 可视化，查看训练效果
# 6. 应用程序，给图识物
# ------------------------------------------------

# ------------------------------------------------
# 6. 应用程序，给图识物
# ------------------------------------------------
# 前向传播执行应用
# 1. 复现模型(前向传播) model = create_model()
# 2. 加载参数  model.load_weights(model_save_path)
# 3. 预测结果 result = model.predict(x)
# ------------------------------------------------

type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ------------------------------------------------
# 模型保存的路径
model_save_path = r"./checkpoint/fashion_model/mnist.ckpt"


# ------------------------------------------------
# 复现模型结构,
# 网络结构必须完全一致, 最好复制训练时候编写构建模型的源代码
# ------------------------------------------------
class MNISTModel(tf.keras.models.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.conv1 = tf.keras.layers.Dense(units=128, activation="relu", use_bias=True)
        self.conv2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.conv1(x)
        y = self.conv2(x)

        return y

# 实例化
model = MNISTModel()
# ------------------------------------------------
# 加载模型权重
model.load_weights(model_save_path)


# ------------------------------------------------
# 模型测试和应用
# ------------------------------------------------
preNum = int(input("input the number of test pictures:"))
for i in range(preNum):
    # ---------------------------------------------------
    # 需要对输入数据进行处理, 满足神经网络模型对输入风格的要求
    # ---------------------------------------------------
    image_path = input("the path of test picture:")

    img = Image.open(image_path)

    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = img.resize((28,28),Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    img_arr = 255 - img_arr

    img_arr = img_arr/255.0

    x_predict = img_arr[tf.newaxis,...]

    result = model.predict(x_predict)
    pred=tf.argmax(result, axis=1)
    print(type[int(pred)])

    plt.pause(1)
    plt.close()
    # ---------------------------------------------------