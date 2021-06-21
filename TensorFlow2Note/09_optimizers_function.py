#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 09_optimizers_function.py
@Function: 神经网络参数优化方法 梯度下降一大类方法：
           iris 鸢尾花分类实现利用不同的优化器, 
           实现对比每一种优化器的 loss 和 accuracy 曲线以及运行时间
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage:
    1. baseline: SGD
    2. baseline + 一阶动量: SGDM，打开注释 SGDM 相关的超参数代码和进行梯度优化的代码
    3. baseline + 二阶动量: Adagrad，打开注释 Adagrad 相关的超参数代码和进行梯度优化的代码
    4. baseline + 二阶动量: RMSProp，打开注释 RMSProp 相关的超参数代码和进行梯度优化的代码
    5. baseline + 一阶动量 + 二阶动量: Adam，打开注释 Adam 相关的超参数代码和进行梯度优化的代码
"""

# ------------------------------------------------
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf

tf.config.set_soft_device_placement(True)


# ------------------------------------------------
# 神经网络参数优化器
# 待优化参数 W；损失函数 loss；学习率 lr；每次迭代一个 batch；t 表示当前 batch 迭代的总次数：
# 1. 计算 t 时刻损失函数关于当前参数的梯度 g(t)
# 2. 计算 t 时刻一阶动量 m(t) 和二阶动量 V(t)
# 3. 计算 t 时刻下降梯度：alph(t) = lr * m(t) / sqrt( V(t) )
# 4. 计算 t+1 时刻参数：w(t+1) = w(t) - alph(t) = w(t) - lr * m(t) / sqrt( V(t) )

# 一阶动量，是与梯度相关的函数
# 二阶动量，是与梯度平方相关的函数
# 不同的优化器区别在于不同的一阶动量和二阶动量
# ------------------------------------------------


# ------------------------------------------------
# SGD, 随机梯度下降，没有一阶动量和二阶动量
# ------------------------------------------------
# m(t) = g(t)；V(t) = 1
# alph(t) = lr * m(t) / sqrt( V(t) ) = lr * g(t)
# w(t+1) = w(t) - alph(t) = w(t) - lr * m(t) / sqrt( V(t) ) = w(t) - lr * g(t)
# ------------------------------------------------


# ------------------------------------------------
# SGDM，在 SGD 基础增加一阶动量
# ------------------------------------------------
# m(t) = beta * m(t-1) * (1-beta)*g(t)；V(t) = 1
# alph(t) = lr * m(t) / sqrt( V(t) ) = lr * ( beta * m(t-1) * (1-beta)*g(t) )
# w(t+1) = w(t) - alph(t) = w(t) - lr * ( beta * m(t-1) * (1-beta)*g(t) )
# ------------------------------------------------


# ------------------------------------------------
# Adagrad, 在 SGD 基础上增加二阶动量
# ------------------------------------------------
# m(t) = g(t)；V(t) = sum( g(t)**2 )
# alph(t) = lr * m(t) / sqrt( V(t) ) = lr * g(t) / sqrt(sum( g(t)**2 ))
# w(t+1) = w(t) - alph(t) = w(t) - lr * g(t) / sqrt(sum( g(t)**2 ))
# ------------------------------------------------


# ------------------------------------------------
# RMSProp, 在 SGD 基础上增加二阶动量
# ------------------------------------------------
# m(t) = g(t)；V(t) = beta * V(t-1) + (1-beta) * ( g(t)**2 )
# alph(t) = lr * m(t) / sqrt( V(t) ) = lr * g(t) / sqrt(beta * V(t-1) + (1-beta) * ( g(t)**2 ))
# w(t+1) = w(t) - alph(t) = w(t) - lr * g(t) / sqrt(beta * V(t-1) + (1-beta) * ( g(t)**2 ))
# ------------------------------------------------


# ------------------------------------------------
# Adam, 同时结合 SGDM 一阶动量和 RMSProp 二阶动量
# ------------------------------------------------
# m(t) = beta1 * m(t-1) + (1-beta1) * g(t)；
# 修正一阶动量的偏差：m(t)_hat = m(t) / (1 - beta1^t)
# V(t) = beta2 * V(t-1) + (1-beta2) * ( g(t)**2 )
# 修正二阶动量的偏差：V(t)_hat = V(t) / (1 - beta2^t)
# alph(t) = lr * m(t) / (1 - beta1^t) / sqrt(V(t) / (1 - beta2^t))
# w(t+1) = w(t) - alph(t) = w(t) - lr * m(t) / (1 - beta1^t) / sqrt(V(t) / (1 - beta2^t))
# ------------------------------------------------


# ------------------------------------------------
# 1. 准备数据
#     - 数据集读入
#     - 数据集乱序
#     - 数据集划分
#     - 有监督的数据对, batch_size
# 2. 搭建模型
#     - 定义神经网络模型
#     - 定义模型所有可训练参数
# 3. 参数优化
#     - 嵌套循环迭代计算
#     - with 结构更新参数
#     - 当前 loss 以及相关信息显示
# 4. 测试效果
#     - 计算当前保存的参数前向计算的准确率
# 5. acc / loss 可视化

# ------------------------------------------------
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化 loss 曲线
# ------------------------------------------------
# 导入数据，分别为输入特征和标签
iris_dataset = datasets.load_iris()
x_data = iris_dataset.data
y_data = iris_dataset.target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（实验结果可重复）
SEED = 42
np.random.seed(SEED)  # 使用相同的 seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(SEED)  # 一次设置只能生效一次 
np.random.shuffle(y_data)

tf.random.set_seed(SEED)

# 将打乱后的数据集分割为训练集和测试集，训练集为前 120 行，测试集为后 30 行
split_test_size = 30
x_train = x_data[:-split_test_size]
y_train = y_data[:-split_test_size]
x_test = x_data[-split_test_size:]
y_test = y_data[-split_test_size:]

# 转换 x 的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices 函数使输入特征和标签值一一对应。（把数据集分批次，每个批次 batch 组数据）
batch_size = 32
train_data_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_data_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


# ------------------------------------------------
# 生成神经网络的参数，
# 4 个输入特征故，输入层为 4 个输入节点；
# 因为 3 分类，故输出层为 3 个神经元
# 用 tf.Variable() 标记参数可训练
# 使用 seed 使每次生成的随机数相同（保证可重复性，在现实使用时不写 seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=SEED))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=SEED))

# 神经网络的复杂度
# 1. 空间复杂度：参数数量
# 2. 时间复杂度：乘加运算的次数

# ------------------------------------------------
lr = 0.1  # 学习率为 0.1
train_loss_results = []  # 将每轮的 loss 记录在此列表中，为后续画 loss 曲线提供数据
test_acc = []  # 将每轮的 acc 记录在此列表中，为后续画 acc 曲线提供数据
loss_all = 0  # 每轮分 4个 step，loss_all 记录四个 step 生成的 4 个 loss 的和
epochs = 500  # 循环 500 轮
steps = np.ceil(len(x_train) / batch_size)
# number of step = ceil( len(train_data) / batch_size ) 
# np.floor() 向下取整
# np.ceil() 向上取整
# np.where() 条件选取
# np.around() 四舍五入

# # ------------------------------------------------
# # 2. baseline + 一阶动量: SGDM，打开 SGDM 相关的超参数代码和进行梯度优化的代码
# # SGDM，在 SGD 基础增加一阶动量; 控制所需要的超参数
# # ------------------------------------------------
# m_w, m_b = 0, 0
# beta = 0.9
# # ------------------------------------------------

# # ------------------------------------------------
# # 3. baseline + 二阶动量: Adagrad，打开 Adagrad 相关的超参数代码和进行梯度优化的代码
# # Adagrad, 在 SGD 基础上增加二阶动量; 控制所需要的超参数
# # ------------------------------------------------
# v_w, v_b = 0, 0
# # ------------------------------------------------

# # ------------------------------------------------
# # 4. baseline + 二阶动量: RMSProp，打开 RMSProp 相关的超参数代码和进行梯度优化的代码
# # RMSProp, 在 SGD 基础上增加二阶动量; 控制所需要的超参数
# # ------------------------------------------------
# v_w, v_b = 0, 0
# beta = 0.9
# # ------------------------------------------------

# ------------------------------------------------
# 5. baseline + 一阶动量 + 二阶动量: Adam，打开 Adam 相关的超参数代码和进行梯度优化的代码
# Adam, 同时结合 SGDM 一阶动量和 RMSProp 二阶动量; 控制所需要的超参数
# ------------------------------------------------
m_w, m_b = 0, 0
v_w, v_b = 0, 0
beta1, beta2 = 0.9, 0.999
delta_w, delta_b = 0, 0
global_step = 0
# ------------------------------------------------

# 训练部分
now_time = time.time()  # 记录开始的时间戳
for epoch in range(epochs):  # 数据集级别的循环，每个 epoch 遍历一次数据集
    for step, (x_train, y_train) in enumerate(train_data_batch):  # batch 级别的循环 ，每个 step 遍历一个 batch
        # ----------------------------------------------------------------
        # 5. baseline + 一阶动量 + 二阶动量: Adam，打开 Adam 相关的超参数代码和进行梯度优化的代码
        # Adam, 同时结合 SGDM 一阶动量和 RMSProp 二阶动量; 控制所需要的超参数
        # ----------------------------------------------------------------
        global_step += 1
        # ----------------------------------------------------------------
        with tf.GradientTape() as tape:  # with 结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出 y 符合概率分布（此操作后与独热码同量级，可相减求 loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算 loss 和 accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数 mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个 step 计算出的 loss 累加，为后续求 loss 平均值提供数据，这样计算的 loss 更准确
        # 计算 loss 对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # # ----------------------------------------------------------------
        # # SGD, 随机梯度下降，没有一阶动量和二阶动量
        # # ----------------------------------------------------------------
        # # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        # w1.assign_sub(lr * grads[0])  # 参数 w1 自更新
        # b1.assign_sub(lr * grads[1])  # 参数 b 自更新
        # # ----------------------------------------------------------------

        # # ----------------------------------------------------------------
        # # 2. baseline + 一阶动量: SGDM，打开 SGDM 相关的超参数代码和进行梯度优化的代码
        # # SGDM，在 SGD 基础增加一阶动量; 控制所需要的超参数
        # # ----------------------------------------------------------------
        # m_w = beta * m_w + (1 - beta) * grads[0]
        # m_b = beta * m_b + (1 - beta) * grads[1]
        # w1.assign_sub(lr * m_w)
        # b1.assign_sub(lr * m_b)
        # # ----------------------------------------------------------------

        # # ----------------------------------------------------------------
        # # 3. baseline + 二阶动量: Adagrad，打开 Adagrad 相关的超参数代码和进行梯度优化的代码
        # # Adagrad, 在 SGD 基础上增加二阶动量
        # # ----------------------------------------------------------------
        # v_w += tf.square(grads[0])
        # v_b += tf.square(grads[1])
        # w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
        # b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))
        # # ----------------------------------------------------------------

        # # ----------------------------------------------------------------
        # # 4. baseline + 二阶动量: RMSProp，打开 RMSProp 相关的超参数代码和进行梯度优化的代码
        # # RMSProp, 在 SGD 基础上增加二阶动量
        # # ----------------------------------------------------------------
        # v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
        # v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
        # w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
        # b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))
        # # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # 5. baseline + 一阶动量 + 二阶动量: Adam，打开 Adam 相关的超参数代码和进行梯度优化的代码
        # Adam, 同时结合 SGDM 一阶动量和 RMSProp 二阶动量
        # ----------------------------------------------------------------
        m_w = beta1 * m_w + (1 - beta1) * grads[0]
        m_b = beta1 * m_b + (1 - beta1) * grads[1]
        v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
        v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])

        m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
        m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
        v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
        v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))

        w1.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction))
        b1.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction))
        # ----------------------------------------------------------------


    # --------------- 完成一次 batch 遍历 ---------------

    # 每个 epoch，打印 loss 信息
    print(f"Epoch {epoch}, loss: {loss_all / steps:.6f}")
    train_loss_results.append(loss_all / steps)  # 将 4 个 step 的 loss 求平均记录在此变量中
    loss_all = 0  # loss_all 归零，为记录下一个 epoch 的 loss 做准备

    # --------------- batch 级别测试 ---------------
    # 测试部分，有两种方案：
    # 其二：每次遍历完成 epoch 就进行测试一次
    # 其一：每次遍历完成 batch 就进行测试一次
    # total_correct 为预测对的样本个数, 
    # total_number 为测试的总样本数，将这两个变量都初始化为 0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_data_batch:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回 y 中最大值的索引，即预测的分类
        # 将 pred 转换为 y_test 的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则 correct=1，否则为 0，将 bool 型的结果转换为 int 型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个 batch 的 correct 数加起来
        correct = tf.reduce_sum(correct)
        # 将所有 batch 中的 correct 数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是 x_test 的行数，shape[0] 返回变量的行数
        total_number += x_test.shape[0]

    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print(f"Test_acc: {acc}")

    # --------------- 完成一次 epoch 遍历 ---------------
total_time = time.time() - now_time  # 记录结束训练的时间戳
print(f"total time: {total_time}")

# ------------------------------------------------
image_path = r"./images"
os.makedirs(image_path, exist_ok=True)

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x 轴变量名称
plt.ylabel('Loss')  # y 轴变量名称
plt.plot(train_loss_results, label="$Loss$", color="red")  # 逐点画出 trian_loss_results 值并连线，连线图标是 Loss
plt.legend()  # 画出曲线图标
# -----------------------------------
# 保存每一种优化器的 loss 图像
# -----------------------------------
# filename = os.path.join(image_path, "iris_loss_SGD.png")
# filename = os.path.join(image_path, "iris_loss_SGDM.png")
# filename = os.path.join(image_path, "iris_loss_Adagrad.png")
# filename = os.path.join(image_path, "iris_loss_RMSProp.png")
filename = os.path.join(image_path, "iris_loss_Adam.png")
plt.savefig(filename, format="png", dpi=500)
# plt.show()  # 画出图像
plt.close()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$", color="red")  # 逐点画出 test_acc 值并连线，连线图标是 Accuracy
plt.legend()
# -----------------------------------
# 保存每一种优化器的 accuracy 图像
# -----------------------------------
# filename = os.path.join(image_path, "iris_accuracy_SGD.png")
# filename = os.path.join(image_path, "iris_accuracy_SGDM.png")
# filename = os.path.join(image_path, "iris_accuracy_Adagrad.png")
# filename = os.path.join(image_path, "iris_accuracy_RMSProp.png")
filename = os.path.join(image_path, "iris_accuracy_Adam.png")
plt.savefig(filename, format="png", dpi=500)
# plt.show()
plt.close()