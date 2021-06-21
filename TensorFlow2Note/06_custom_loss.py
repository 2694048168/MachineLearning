#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 06_custom_loss.py
@Function: 根据需要自定义损失函数的实现
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
# 自定义损失函数，不对称贡献，分段损失函数
# 成本 1 元， 利润 99 元；经常遇到这种贡献不对称的现象
# 成本很低，利润很高，希望多预测些，生成模型系数大于 1，往多了预测
# ------------------------------------------------
SEED = 42
# 自定义损失函数，均方误差默认对于正样本和负样本的距离(贡献)是对称的
# 有时候需要不对称的，分段函数，自定义
COST = 1
PROFIT = 99

rdm = np.random.RandomState(SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=SEED))

epoch = 10000
lr = 0.002

for epoch in range(epoch):
    # with 结构记录梯度信息
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        # 自定义损失函数，分段函数，不对称的贡献
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

    grads = tape.gradient(loss, w1)
    # 更新参数
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print(f"After {epoch} training steps, w1 is: {w1.numpy()}")

print("---------------------------")
print(f"Finally w1 is: {w1.numpy()}")