#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 18_cifar10_baseline.py
@Function: TensroFlow2 API tf.keras 实现卷积神经网络对 CIFAR10 分类
@Linking: https://www.deeplearningbook.org/contents/mlp.html#pf25
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 18_cifar10_baseline.py 
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


# ----------------------------------------------------------
# 卷积神经网络(Convolutional Neural Network, CNN)实现离散数据分类
# ----------------------------------------------------------
# 1. 卷积计算过程
# 2. 感受野和有效感受野 Receptive Field
# 3. 全零填充 Zero Padding
# 4. TensorFlow 描述卷积计算层
# 5. 批标准化(Batch Normalization, BN)
# 6. 池化 Pooling
# 7. 舍弃 Dropout
# 8. 卷积神经网络
# 9. CIFAR10 and CIFAR100 数据集
# 10. 卷积神经网络搭建架构
# 11. LeNet; AlexNet; VGGNet; IceptionNet, ResNet
# ----------------------------------------------------------
# ----------------------------------------------------------
# 1. 卷积计算过程
# ----------------------------------------------------------
# 全连接网络,参数太多了, 待优化的参数过多容易导致模型过拟合
# 卷积神经网络, 对原始图像做特征提取(很多次的特征提取)
# 卷积计算是一种有效的图像特征提取方法, 卷积核-卷积步长-深度匹配
# 输入特征图的深度(channels)决定了当前层卷积核的深度(卷积核数量)
# 当前卷积核的个数,决定了当前层输出特征图的深度
# ----------------------------------------------------------
# ----------------------------------------------------------
# 2. Receptive Field
# 指卷积神经网络各输出特征图中的每个像素点,在原始输入图像上映射区域的大小
# 感受野大小一致(多个小卷积核等效一个大卷积核), 待优化的参数量不同, 同时乘加计算量不同
# ----------------------------------------------------------
# 3. 全零填充 Zero Padding
# 保证特征图大小不变, 有卷积计算公式可以计算输入和输出维度, 计算结果向上取整
# padding="same" 表示使用 zero padding
# padding="valid" 表示不使用使用 zero padding 
# ----------------------------------------------------------
# tf.keras.layers.Conv2D(
#     filters, 
#     kernel_size, 
#     strides=(1, 1), 
#     padding='valid',
#     data_format=None, 
#     dilation_rate=(1, 1), 
#     groups=1, 
#     activation=None,  # 有 BN 层,则不需要激活函数
#     use_bias=True, 
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros', 
#     kernel_regularizer=None,
#     bias_regularizer=None, 
#     activity_regularizer=None, 
#     kernel_constraint=None,
#     bias_constraint=None, 
#     **kwargs
# )
# ----------------------------------------------------------
# 5. 批标准化(Batch Normalization, BN)
# 标准化: 使数据符合 0 均值, 1 标准差的分布
# 批标准化: 对小批量数据(batch),做标准化处理
# 减去均值, 除以标准差
# BN 操作,将原本偏移的特征数据,重新拉回到 0 均值, 使进入激活函数的数据分布在激活函数的线性区域,
# 使得输入数据的微小变化, 更明显的体现到激活函数的输出, 提升激活函数对输入数据的区分力,
# 但是这样使得特征数据分布完全满足标准正态分布, 集中在激活函数中心的线性区域, 使得激活函数丧失了非线性特性
# 因此 BN 操作中为每一个卷积核引入两个可训练的参数: 缩放因子和偏移因子
# 通过优化这两个参数, 优化了特征数据分布的宽窄和偏移量, 保证非线性表达能力
# BN 层, 通常在卷积层和激活层之间 Conv-BN-Activation
# ----------------------------------------------------------
# tf.keras.layers.BatchNormalization(
#     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
#     beta_initializer='zeros', gamma_initializer='ones',
#     moving_mean_initializer='zeros',
#     moving_variance_initializer='ones', beta_regularizer=None,
#     gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs
# )
# ----------------------------------------------------------
# 6. 池化 Pooling
# 减少特征数据量, 最大池化提取图像纹理信息; 均值池化保留背景特征
# tf.keras.layers.MaxPool2D(
#     pool_size=(2, 2), strides=None, padding='valid', data_format=None,
#     **kwargs
# )
# ----------------------------------------------------------
# tf.keras.layers.AveragePooling2D(
#     pool_size=(2, 2), strides=None, padding='valid', data_format=None,
#     **kwargs
# )
# ----------------------------------------------------------
# 7. 舍弃 Dropout
# 在训练过程中,将神经元按照一定概率暂时舍弃; 测试时,被舍弃的神经元恢复
# ----------------------------------------------------------
# CNN baseline
# ----------------------------------------------------------
# 卷积神经网络主要模块:卷积-批标准化-激活-池化-舍弃-全连接
# 卷积就是特征提取器,就是 CBAPD; 卷积神经网络的八股套路
# Convolutional-BN-Activation-Pooling-Dropout
# ----------------------------------------------------------


# ----------------------------------------------------------
# 加载数据集
# ----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# ----------------------------------------------------------
# 建立卷积神经网络模型
# ----------------------------------------------------------
class Baseline(tf.keras.models.Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=6, 
                                            kernel_size=(5, 5),
                                            strides=(1, 1),
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("relu")
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.drop1 = tf.keras.layers.Dropout(rate=0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.drop2 = tf.keras.layers.Dropout(rate=0.2)
        self.fc2 = tf.keras.layers.Dense(units=10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop2(x)
        y = self.fc2(x)

        return y

# 实例化模型
model = Baseline()

# ----------------------------------------------------------
# 编译模型
# ----------------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ----------------------------------------------------------
# 断点续练模型
# ----------------------------------------------------------
checkpoint_save_path = r"./checkpoint/cifar_baseline/baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         save_best_only=True)

# ----------------------------------------------------------
# 训练模型
# ----------------------------------------------------------
history = model.fit(x_train, 
                    y_train, 
                    batch_size=32, 
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
weight_txt = os.path.join(csv_file, "cifar_baseline.txt")
with open(weight_txt, "w") as file:
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')


# ----------------------------------------------------------
# 可视化模型的 acc 和 loss
# ----------------------------------------------------------
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
plt.close()