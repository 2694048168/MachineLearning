#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 26_rnn_embedding.py
@Function: TensroFlow2 API tf.keras 实现 RNN 对字母进行预测
            使用 Embedding 编码方法
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 26_rnn_embedding.py 
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
input_word = "abcde"
word_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}  # 单词映射到数值id的词典

x_train = [word_to_id['a'], word_to_id['b'], word_to_id['c'], word_to_id['d'], word_to_id['e']]
y_train = [word_to_id['b'], word_to_id['c'], word_to_id['d'], word_to_id['e'], word_to_id['a']]

SEED = 42
np.random.seed(seed=SEED)
np.random.shuffle(x_train)
np.random.seed(seed=SEED)
np.random.shuffle(y_train)

tf.random.set_seed(seed=SEED)

# 使 x_train 符合 Embedding 输入要求：[送入样本数， 循环核时间展开步数] ，
# 此处整个数据集送入所以送入，送入样本数为 len(x_train)；
# 输入 1 个字母出结果，循环核时间展开步数为 1
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)
# ----------------------------------------------------------
# ------------------------------------------------
# one-hot 编码：数据量大，过于稀疏，映射之间是独立的，没有表现出关联性
# Embedding 编码方法
# Embedding 是一种单词编码方法，用低维向量实现编码，
# 这种编码通过神经网络训练优化，能表达出单词之间的关联性
# ------------------------------------------------
# tf.keras.layers.Embedding(词汇表大小， 编码维度)
# 编码维度就是用几个数字表达一个单词
# 进入 Embedding 时候， x_train 维度：[送入样本数，循环核时间展开步数]
# ------------------------------------------------
# ----------------------------------------------------------


# ----------------------------------------------------------
# 建立 RNN 模型, 提取时间序列信息
# ----------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5, 2),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(units=5, activation="softmax")
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
checkpoint_save_path = r"./checkpoint/rnn_embedding/rnn_embedding.ckpt"
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
weight_txt = os.path.join(csv_file, "rnn_embedding.txt")
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
    alphabet = [word_to_id[alphabet1]]
    # 使 alphabet 符合 Embedding 输入要求：[送入样本数， 循环核时间展开步数]。
    # 此处验证效果送入了 1 个样本，送入样本数为 1；
    # 输入 1 个字母出结果，循环核时间展开步数为 1
    alphabet = np.reshape(alphabet, (1, 1))
    result = model.predict([alphabet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])
# ------------------------------------------------