#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 02_tensor_tf.py
@Function: 张量 Tensor tensroflow2 和 Numpy 常用函数
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
print("------------------------------------------------")
# OS 默认 tf.int32 ?
tensor_a = tf.constant([1, 5], dtype=tf.int64)
print("tensor_a:", tensor_a)
print("tensor_a.dtype:", tensor_a.dtype)
print("tensor_a.shape:", tensor_a.shape)


# ------------------------------------------------
print("------------------------------------------------")
ndarray_a = np.arange(0, 5)
# numpy 数据类型转化为 tensor
tensor_b = tf.convert_to_tensor(ndarray_a, dtype=tf.int32)
print("ndarray_a:", ndarray_a)
print("tensor_b:", tensor_b)


# ------------------------------------------------
print("------------------------------------------------")
tensor_zeros = tf.zeros([2, 3])
tensor_ones = tf.ones(4)
# 创建指定值的张量
tensor_fill_value = tf.fill([2, 2], 9)
print("tensor_zeros:", tensor_zeros)
print("tensor_ones:", tensor_ones)
print("tensor_fill_value:", tensor_fill_value)


# ------------------------------------------------
print("------------------------------------------------")
# 正态分布的随机数，默认标准正态分布 (0，1)
tensor_normal = tf.random.normal([2, 2], mean=0.5, stddev=1)
# 分布更加靠近均值，两倍标准差之内，数据向均值集中
# (mean - 2*std, mean + 2*std)
tensor_trunc_normal = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("tensor_normal:", tensor_normal)
print("tensor_trunc_normal:", tensor_trunc_normal)

# 均匀分布 [minval, maxval), 前闭后开区间
tensor_uniform = tf.random.uniform([2, 2], minval=0, maxval=1)
print("tensor_uniform:", tensor_uniform)


# ------------------------------------------------
print("------------------------------------------------")
# tf.Variable() 变量标记为 "可训练"，在反向传播中自动记录梯度信息，可更新
# 初始化权重
W = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
print(W)


# ------------------------------------------------
print("------------------------------------------------")
# 返回一个 [0，1) 之间的随机数
SEED = 42
rdm = np.random.RandomState(seed=SEED)
rdm_a = rdm.rand()
# 维度
rdm_b = rdm.rand(2, 3)
print("a:", rdm_a)
print("b:", rdm_b)


# ------------------------------------------------
print("------------------------------------------------")
ndarray_a = np.array([1, 2, 3])
ndarray_b = np.array([4, 5, 6])
# 两个数组按照垂直方向叠加
ndarray_c = np.vstack((ndarray_a, ndarray_b))
print("c:\n", ndarray_c)

# 两个数组按照水平方向叠加
print(np.hstack((ndarray_a, ndarray_b)))


# ------------------------------------------------
print("------------------------------------------------")
# 三种函数组合使用生成网格坐标点

# 生成等间隔数值点
x, y = np.mgrid[1:3:1, 2:4:0.5]
# 将 x, y 拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[x.ravel(), y.ravel()]
print("x:\n", x)
print("y:\n", y)
print("x.ravel():\n", x.ravel())
print("y.ravel():\n", y.ravel())
print('grid:\n', grid)
