#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 04_iris_demo.py
@Function: iris 鸢尾花分类 tensroflow2 实现 MLP
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf

tf.config.set_soft_device_placement(True)


# ------------------------------------------------
# 对 iris 数据集进行探索
# iris_dataset = datasets.load_iris()
# x_data = iris_dataset.data  # .data 返回 iris 数据集所有输入特征
# y_data = iris_dataset.target  # .target 返回 iris 数据集所有标签
# print("x_data from datasets: \n", x_data)
# print("y_data from datasets: \n", y_data)

# x_data = pd.DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']) # 为表格增加行索引（左侧）和列标签（上方）
# pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
# print("x_data add index: \n", x_data)

# x_data['类别'] = y_data  # 新加一列，列标签为‘类别’，数据为 y_data
# print("x_data add a column: \n", x_data)
# # 类型维度不确定时，建议用 print 函数打印出来确认效果
# print(x_data.shape)


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
learning_rate = 0.1  # 学习率为 0.1
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

# 训练部分
for epoch in range(epochs):  # 数据集级别的循环，每个 epoch 遍历一次数据集
    for step, (x_train, y_train) in enumerate(train_data_batch):  # batch 级别的循环 ，每个 step 遍历一个 batch
        with tf.GradientTape() as tape:  # with 结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出 y 符合概率分布（此操作后与独热码同量级，可相减求 loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算 loss 和 accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数 mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个 step 计算出的 loss 累加，为后续求 loss 平均值提供数据，这样计算的 loss 更准确
        # 计算 loss 对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(learning_rate * grads[0])  # 参数 w1 自更新
        b1.assign_sub(learning_rate * grads[1])  # 参数 b 自更新

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


# ------------------------------------------------
image_path = r"./images"
os.makedirs(image_path, exist_ok=True)

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x 轴变量名称
plt.ylabel('Loss')  # y 轴变量名称
plt.plot(train_loss_results, label="$Loss$", color="red")  # 逐点画出 trian_loss_results 值并连线，连线图标是 Loss
plt.legend()  # 画出曲线图标
filename = os.path.join(image_path, "iris_loss.png")
plt.savefig(filename, format="png", dpi=500)
# plt.show()  # 画出图像s
plt.close()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$", color="red")  # 逐点画出 test_acc 值并连线，连线图标是 Accuracy
plt.legend()
filename = os.path.join(image_path, "iris_accuracy.png")
plt.savefig(filename, format="png", dpi=500)
# plt.show()
plt.close()