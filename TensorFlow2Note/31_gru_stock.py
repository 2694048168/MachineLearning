#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 31_gru_stock.py
@Function: TensroFlow2 API tf.keras 实现 GRU 对股票的开盘价进行预测
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 31_gru_stock.py 
"""

# ------------------------------------------------
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

np.set_printoptions(threshold=np.inf)
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
# ---------------------------------------------------
# RNN 实现连续数据的预测(金融股票数据)
# ---------------------------------------------------
# 循环神经网络
# 1. 循环核
# 2. 循环核时间步展开
# 3. 循环计算层
# 4. TensorFlOW 描述循环计算层
# 5. 循环计算过程
# ---------------------------------------------------
# 实践：ABCDE 字母预测
# 实现功能：输入 a 预测出 b；输入 b 预测出 c；输入 c 预测出 d；输入 d 预测出 e；输入 e 预测出 a
# 1. one-hot (词向量空间实现的最简单方法)
# 2. Embedding
# ---------------------------------------------------
# 实践：股票预测
# 1. RNN
# 2. LSTM
# 3. GRU
# ---------------------------------------------------
# 卷积核：参数空间共享，卷积层提取空间信息
# ---------------------------------------------------
# 数据具有时序性，上下文信息；
# 通过脑记忆体提取历史数据的特征，预测最有可能出现的情况
# 循环核：参数时间共享，循环层提取时间信息
# 循环核具有记忆力，通过对不同时刻的参数共享，实现对时间序列的信息提取
# ---------------------------------------------------
# 4. TensorFlOW 描述循环计算层
# tf.keras.layers.SimpleRNN(记忆体个数， activation="激活函数", return_sequences=是否每个时刻输出 h(t) 到下一层)
# activation="tanh"; 默认值
# activation="False"; 默认值，仅在最后时间输出 h(t)
# activation="True"; 默认值，各个时间输出 h(t)
# ---------------------------------------------------
# RNN层期待维度 x_train 维度：[送入样本数，循环核时间展开步数，每个时间步输入特征个数]
# ---------------------------------------------------

# ---------------------------------------------------
# GRU 计算过程
# 更新们
# 重置门
# 记忆体
# 候选隐藏层
# ---------------------------------------------------
# TensorFLow 描述
# tf.keras.layers.GRU(记忆体个数，return_sequences=是否返回输出)
# ---------------------------------------------------


# ----------------------------------------------------------
# 加载数据集
# ----------------------------------------------------------
csv_file = r"./csv_file"
moutai = pd.read_csv(os.path.join(csv_file, './moutai.csv'))  # 读取股票文件
# 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
training_set = moutai.iloc[0:2475 - 300, 2:3].values  
test_set = moutai.iloc[2475 - 300:, 2:3].values  # 后300天的开盘价作为测试集

# 归一化, scikit-learn 
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化

x_train = []
y_train = []

x_test = []
y_test = []

# 测试集：csv表格中前 2475-300 天数据
# 利用 for 循环，遍历整个训练集，
# 提取训练集中连续 60 天的开盘价作为输入特征 x_train，第61天的数据作为标签，
# for 循环共构建 2475-300-60 组数据
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# 对训练集进行打乱
SEED = 42
np.random.seed(seed=SEED)
np.random.shuffle(x_train)
np.random.seed(seed=SEED)
np.random.shuffle(y_train)

tf.random.set_seed(seed=SEED)

# 将训练集由 list 格式变为 array 格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使 x_train 符合 RNN 输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
# 此处整个数据集送入，送入样本数为 x_train.shape[0] 即 2475-300 组数据；
# 输入 60 个开盘价，预测出第 61 天的开盘价，循环核时间展开步数为 60; 
# 每个时间步送入的特征是某一天的开盘价，只有 1 个数据，故每个时间步输入特征个数为 1
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
# 测试集：csv 表格中后 300 天数据
# 利用 for 循环，遍历整个测试集，提取测试集中连续 60 天的开盘价作为输入特征 x_train，
# 第 61 天的数据作为标签，for 循环共构建 300-60 组数据
for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])

# 测试集变 array 并 reshape 为符合 RNN 输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))


# ----------------------------------------------------------
# 建立 RNN 模型, 提取时间序列信息
# ----------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.GRU(80, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GRU(100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


# ----------------------------------------------------------
# 编译模型
# ----------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_squared_error')  # 损失函数用均方误差
# 该应用只观测 loss 数值，不观测准确率，所以删去 metrics 选项，一会在每个 epoch 迭代显示时只显示 loss 值


# ----------------------------------------------------------
# 断点续练模型
# ----------------------------------------------------------
checkpoint_save_path = r"./checkpoint/GRU_stock/GRU_stock.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

# 由于 fit 没有给出测试集，不计算测试集准确率，根据 loss，保存最优模型
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         monitor="val_loss")


# ----------------------------------------------------------
# 训练模型
# ----------------------------------------------------------
history = model.fit(x_train, 
                    y_train, 
                    batch_size=64, 
                    epochs=50,
                    validation_data=(x_test, y_test), 
                    validation_freq=1,
                    callbacks=[checkpoint_callback],
                    verbose=2)
                 

# ----------------------------------------------------------
# 查看模型参数和结构
# ----------------------------------------------------------
model.summary()


# ----------------------------------------------------------
# 将训练好的模型参数保存到文本文件中
# ----------------------------------------------------------
# print(model.trainable_variables)
csv_file = r"./csv_file"
os.makedirs(csv_file, exist_ok=True)
weight_txt = os.path.join(csv_file, "GRU_stock.txt")
with open(weight_txt, "w") as file:
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')


# ----------------------------------------------------------
# 可视化模型的 acc 和 loss
# ----------------------------------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.close()


# ----------------------------------------------------------
# 使用训练好的模型进行预测 predict
# ----------------------------------------------------------
# 测试集输入模型进行预测
predicted_stock_price = model.predict(x_test)
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(test_set[60:])
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()
plt.close()

# ----------------------------------------------------------
# ------------- evaluate -------------
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
# ----------------------------------------------------------