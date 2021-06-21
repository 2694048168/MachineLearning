#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 11_mnist_keras.py
@Function: MNIST 手写体数值识别 tensroflow2 API tf.keras 实现
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# # ------------------------------------------------
# # 参数都是可以以 字符串形式给出；或者使用函数形式给出
# # ------------------------------------------------
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


# ------------------------------------------------
# 使用类继承的形式编写自定义的网络模型
# 前向计算
# 后向传播
# ------------------------------------------------
class MnistModel(tf.keras.models.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.conv1 = tf.keras.layers.Dense(128, activation='relu')
        self.conv2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.conv1(x)
        y = self.conv2(x)

        return y

# 实例化
model = MnistModel()
# ------------------------------------------------


# ------------------------------------------------
# from_logits=True 询问是否是原始输出(没有经过概率分布的输出)
# 输出经过概率分布 (softmax)，False
# 输出没有经过概率分布 (softmax)，True
# ------------------------------------------------
# metrics 可选
# accuracy：y_hat and y_pred 都是数值，如 y_hat=[1] y_pred=[1]
# categorical_accuracy: y_hat and y_pred 都是独热编码格式 (概率分布)
# sparse_categorical_accuracy: y_hat 是数值， y_pred 是独热编码格式 (概率分布)
# ------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ------------------------------------------------
# validation_freq 多少次 epoch 进行测试一次
# ------------------------------------------------
model.fit(x_train, 
          y_train, 
          batch_size=32, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          validation_freq=1,
          verbose=2)

# ------------------------------------------------
model.summary()
# ------------------------------------------------
# 可视化网络结构
# image_path = r"./images"
# os.makedirs(image_path, exist_ok=True)

# tf.keras.utils.plot_model(model, 
#                           to_file=os.path.join(image_path, "mnist_mlp.png"), 
#                           show_shapes=True, dpi=500)