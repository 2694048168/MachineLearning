#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 07_cross_entropy.py
@Function: 度量两个分布之间的距离，交叉熵；KL散度; softmax
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

tf.config.set_soft_device_placement(True)


# ------------------------------------------------
# tensorflow 计算交叉熵
loss_cross_entropy1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_cross_entropy2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print("loss_cross_entropy1:", loss_cross_entropy1.numpy())
print("loss_cross_entropy2:", loss_cross_entropy2.numpy())

# ------------------------------------------------
# 交叉熵损失函数 cross entropy
# 度量两个分布之间的距离，KL散度
# 严格的数学推导：熵的定义；交叉熵的定义；KL散度定义；以及三者之间的关系
# ------------------------------------------------


# ------------------------------------------------
print("----------------------------------------")
# softmax 与交叉熵损失函数的结合
# label one-hot code
y_ = np.array([[1, 0, 0],
               [0, 1, 0], 
               [0, 0, 1], 
               [1, 0, 0], 
               [0, 1, 0]])

# 预测值利用 softmax 函数转化为概率值   
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_crossentropy(y_, y_pro)

# 同时计算概念分布和交叉熵，一次性计算
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)

print('分步计算的结果:\n', loss_ce1)
print('结合计算的结果:\n', loss_ce2)