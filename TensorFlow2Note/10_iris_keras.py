#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 10_iris_keras.py
@Function: iris 鸢尾花分类 tensroflow2 API tf.keras 实现
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from sklearn import datasets
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
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

SEED = 42
np.random.seed(SEED)
np.random.shuffle(x_train)
np.random.seed(SEED)
np.random.shuffle(y_train)

tf.random.set_seed(SEED)

# # ------------------------------------------------
# # 参数都是可以以 字符串形式给出；或者使用函数形式给出
# # ------------------------------------------------
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
# ])


# ------------------------------------------------
# 使用类继承的形式编写自定义的网络模型
# 前向计算
# 后向传播
# ------------------------------------------------
class IrisModel(tf.keras.Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)

        return y

# 实例化
model = IrisModel()
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
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ------------------------------------------------
# validation_freq 多少次 epoch 进行测试一次
# ------------------------------------------------
model.fit(x_train, 
          y_train, 
          batch_size=32, 
          epochs=500, 
          validation_split=0.2, 
          validation_freq=20,
          verbose=2)

# ------------------------------------------------
model.summary()
# ------------------------------------------------
# 可视化网络结构
# image_path = r"./images"
# os.makedirs(image_path, exist_ok=True)

# tf.keras.utils.plot_model(model, 
#                           to_file=os.path.join(image_path, "iris_mlp.png"), 
#                           show_shapes=True, dpi=500)