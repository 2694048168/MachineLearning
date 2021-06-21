#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 03_function_utils.py
@Function: tensroflow2 常使用的功能性函数
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
x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)
# 强制 tensor 进行类型转换
x2 = tf.cast(x1, tf.int32)
print("x2", x2)

# 计算张量的最值，不指定则计算张量的所有元素
# axis 操作张量的维度，矩阵而言 axis=0， 对行操作（经度）；axis=1，对列操作（维度）
print("minimum of x2：", tf.reduce_min(x2))
print("maxmum of x2:", tf.reduce_max(x2))


# ------------------------------------------------
print("------------------------------------------------")
x = tf.constant([[1, 2, 3], 
                 [2, 2, 3]], dtype=tf.float32)
print("x:", x)
print("mean of x:", tf.reduce_mean(x))  # 求 x 中所有数的均值
print("mean of x:", tf.reduce_mean(x, axis=0))  # 求 x 中 axis=0 维度的均值
print("sum of x:", tf.reduce_sum(x, axis=1))  # 求每一行的和


# ------------------------------------------------
print("------------------------------------------------")
a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print("a:", a)
print("b:", b)
# 两个 tensor 进行逐元素 四则运算, 自动类型转换
print("a+b:", tf.add(a, b))
print("a-b:", tf.subtract(a, b))
print("a*b:", tf.multiply(a, b))
print("b/a:", tf.divide(b, a))

# element-wise operator
tensor_a = tf.fill([1, 2], 3.)
print("tensor_a:", tensor_a)
print("tensor_a 的三次方方:", tf.pow(tensor_a, 3))
print("tensor_a 的平方:", tf.square(tensor_a))
print("tensor_a 的开方:", tf.sqrt(tensor_a))


# ------------------------------------------------
print("------------------------------------------------")
# 矩阵乘法
tensor_ones = tf.ones([3, 2])
tensor_fill = tf.fill([2, 3], 3.)
print("tensor_ones:", tensor_ones)
print("tensor_fill:", tensor_fill)
print("tensor_ones * tensor_fill:", tf.matmul(tensor_ones, tensor_fill))


# ------------------------------------------------
print("------------------------------------------------")
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
# 特征和标签配对 (numpy格式和 tensor格式都可以)
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)
    break

for feature, label in dataset:
    print(feature.numpy())
    print(label.numpy())
    break


# ------------------------------------------------
print("------------------------------------------------")
# with 结构记录计算过程, gradient 求解张量的梯度
with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))
    y = tf.pow(x, 2)
grad = tape.gradient(y, x)
print(grad)


# ------------------------------------------------
print("------------------------------------------------")
seq = ['one', 'two', 'three']
# 枚举, 有索引号, 任何可迭代的对象, python 常用的函数
for i, element in enumerate(seq):
    print(i, element)


# ------------------------------------------------
print("------------------------------------------------")
classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为 0，最大为 2
# 独热码, one-hot coding
output = tf.one_hot(labels, depth=classes)
print(labels)
print("result of labels1:", output)


# ------------------------------------------------
print("------------------------------------------------")
x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])
b1 = tf.constant([2.52, -3.1, 5.62])
# 前向传播
# [1, 3] = [1, 4] @ [4, 3] + [3,]
y = tf.matmul(x1, w1) + b1
print("x1.shape:", x1.shape)
print("w1.shape:", w1.shape)
print("b1.shape:", b1.shape)
print("y.shape:", y.shape)
print("y:", y)

# 可将输出结果 y 转化为概率值
y_dim = tf.squeeze(y)  # 去掉 y 中纬度为 1（观察 y_dim 与  y 效果对比）
y_pro = tf.nn.softmax(y_dim)  # 使 y_dim 符合概率分布，输出为概率值了
# 请观察打印出的 shape
print("y_dim:", y_dim)
print("y_pro:", y_pro)
print("sum of y_pro:", tf.reduce_sum(y_pro))  # 概念之和为 1


# ------------------------------------------------
print("------------------------------------------------")
x = tf.Variable(4)
# 参数自更新, 自减
x.assign_sub(1)
print("x:", x)  # 4-1=3


# ------------------------------------------------
print("------------------------------------------------")
test = np.array([[1, 2, 3], 
                 [2, 3, 4], 
                 [5, 4, 3], 
                 [8, 7, 2]])
print("test:\n", test)
# 返回张量指定轴维度的最大值索引
print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的索引
print("每一列的最大值的索引：", tf.argmax(test))  # 返回每一列最大值的索引
print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的索引


# ------------------------------------------------
print("------------------------------------------------")
expression_a = tf.constant([1, 2, 3, 1, 1])
expression_b = tf.constant([0, 1, 3, 4, 5])
# 条件语句 True 返回 A，否则返回 B
expression_c = tf.where(tf.greater(expression_a, expression_b), expression_a, expression_b) 
# 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
print("a：", expression_a)
print("b：", expression_b)
print("c：", expression_c)