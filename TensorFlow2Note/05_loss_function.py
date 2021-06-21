#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 05_loss_function.py
@Function: 神经网络常用的损失函数的 tensroflow2 实现
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
# 激活函数 Activation function
# ------------------------------------------------
# Sigmoid function
# f(x) = 1 / (1 + exp(-x))
# tf.nn.sigmoid(x)
# 1. 根据其导数曲线图可知，容易造成梯度消失
# 2. 输出非 0 均值，收敛慢
# 3. 幂运算复杂，训练时间长

# Tanh function
# f(x) = (1 - exp(-2x)) / (1 + exp(-2x))
# tf.math.tanh(x)
# 1. 根据其导数曲线图可知，容易造成梯度消失
# 2. 输出非 0 均值，收敛慢
# 3. 幂运算复杂，训练时间长

# Relu function
# f(x) = max(x, 0)
# tf.nn.relu(x)
# 1. 在正区间解决梯度消失现象
# 2. 只需要判断输入是否大于 0，计算速度块
# 3. 收敛速度远快于 sigmoid 和 tanh

# 1. 输出非 0 均值，收敛慢
# 2. Dead Relu 问题：某些神经元可能永远不被激活，导致对应的参数永远不能更新
# 3. 主要原因是参数中 负数 过多

# Leaky Relu function
# f(x) = max(ax, x)
# tf.nn.leaky_relu(x)
# 1. 在正区间解决梯度消失现象
# 2. 只需要判断输入是否大于 0，计算速度块
# 3. 收敛速度远快于 sigmoid 和 tanh

# 1. 输出非 0 均值，收敛慢
# 2. 解决 Dead Relu 现象
# 3. 主要原因是参数中 负数 有一个线性变换

# 激活函数建议
# 1. 首选 ReLU 激活函数
# 2. 学习率设置较小值
# 3. 输入特征标准化，即让输入特征满足以 0 为均值，1 为标准差的正态分布
# 4. 初始参数中心化，即让随机生成的参数满足以 0 为均值，sqrt(2 / 当前输入特征的个数) 为标准差的正态分布
# ------------------------------------------------


# ------------------------------------------------
SEED = 42

# 模拟一个数据集，使用均方误差作为损失函数
# y = x1 + x2, 即最优的 w1[0] = w1[1] = 1
rdm = np.random.RandomState(seed=SEED)  # 生成[0,1)之间的随机数
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)

# 输入特征，x1; x2; 输出结果，y
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=SEED))

epochs = 15000
lr = 0.002

for epoch in range(epochs):
    # with 结构记录梯度信息
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        # 均方误差计算
        loss_mse = tf.reduce_mean(tf.square(y_ - y))

    grads = tape.gradient(loss_mse, w1)
    # 参数更新
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print(f"After {epoch} training steps, w1 is: {w1.numpy()}")

print("---------------------------")
print(f"Finally w1 is: {w1.numpy()}")