#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 27_rnn_embedding_4to1.py
@Function: TensroFlow2 API tf.keras 实现 RNN 对字母进行预测
            使用 Embdding 编码方法
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 27_rnn_embedding_4to1.py 
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
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


# ----------------------------------------------------------
# 加载数据集
# ----------------------------------------------------------
input_word = "abcdefghijklmnopqrstuvwxyz"
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
           'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
           'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
           'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
           'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}  # 单词映射到数值id的词典

training_set_scaled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                       21, 22, 23, 24, 25]

x_train = []
y_train = []

for i in range(4, 26):
    x_train.append(training_set_scaled[i - 4:i])
    y_train.append(training_set_scaled[i])


SEED = 42
np.random.seed(seed=SEED)
np.random.shuffle(x_train)
np.random.seed(seed=SEED)
np.random.shuffle(y_train)

tf.random.set_seed(seed=SEED)
# ----------------------------------------------------------
# 使 x_train 符合 Embedding 输入要求：[送入样本数， 循环核时间展开步数] ，
# 此处整个数据集送入所以送入，送入样本数为 len(x_train)；
# 输入 4 个字母出结果，循环核时间展开步数为 4
x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)
# ----------------------------------------------------------


# ----------------------------------------------------------
# 建立 RNN 模型, 提取时间序列信息
# ----------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(26, 2),
    tf.keras.layers.SimpleRNN(10),
    tf.keras.layers.Dense(units=26, activation="softmax")
])


# ----------------------------------------------------------
# 编译模型
# ----------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])


# ----------------------------------------------------------
# 断点续练模型
# ----------------------------------------------------------
checkpoint_save_path = r"./checkpoint/rnn_embedding_4to1/rnn_embedding_4to1.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

# 由于 fit 没有给出测试集，不计算测试集准确率，根据 loss，保存最优模型
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         monitor="loss")


# ----------------------------------------------------------
# 训练模型
# ----------------------------------------------------------
history = model.fit(x_train, 
                    y_train, 
                    batch_size=32, 
                    epochs=100, 
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
weight_txt = os.path.join(csv_file, "rnn_embedding_4to1.txt")
with open(weight_txt, "w") as file:
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')


# ----------------------------------------------------------
# 可视化模型的 acc 和 loss
# ----------------------------------------------------------
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.show()
plt.close()


# ----------------------------------------------------------
# 使用训练好的模型进行预测 predict
# ----------------------------------------------------------
preNum = int(input("input the number of test alphabet:"))
for i in range(preNum):
    alphabet1 = input("input test alphabet:")
    alphabet = [w_to_id[a] for a in alphabet1]
    # 使 alphabet 符合 Embedding 输入要求：[送入样本数， 时间展开步数]
    # 此处验证效果送入了 1 个样本，送入样本数为 1；
    # 输入 4 个字母出结果，循环核时间展开步数为 4
    alphabet = np.reshape(alphabet, (1, 4))
    result = model.predict([alphabet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])
# ------------------------------------------------