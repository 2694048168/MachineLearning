#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 23_cifar10_resnet.py
@Function: TensroFlow2 API tf.keras 实现 ResNet 对 CIFAR10 分类
@Paper: Deep Residual Learning for Image Recognition
@Linking: https://arxiv.org/abs/1512.03385
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 23_cifar10_resnet.py 
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
# 加载数据集
# ----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# ----------------------------------------------------------
# 建立 ResNet 模型
# ----------------------------------------------------------

# ----------------------------------------------------------
# ResNet 结构快: shortcut(恒等映射); 特征图匹配
# Inception block 中的 "+" 是指沿着深度方向叠加 (千层蛋糕层数叠加)
# ResNet block 中的 "+" 是指特征图对应元素值相加 (矩阵值相加)
# 当特征图的大小([H, W, C])相等时候, 是直接进行相加, 实线表示;
# 当特征图的大小([H, W, C])不一致时候,  借助 1*1 卷积来调整恒等映射的维度匹配, 虚线表示
# 1*1 卷积操作可以通过步长改变特征图的尺寸[H, W], 通过卷积核个数来改变特征图的深度[C]
# 调整 batch_size , 使得 GPU 负载在 80%-90%
# ----------------------------------------------------------
class ResnetBlock(tf.keras.models.Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')

        self.c2 = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = tf.keras.layers.BatchNormalization()

        # residual_path 为 True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = tf.keras.layers.BatchNormalization()
        
        self.a2 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out

# ----------------------------------------------------------
# ----------------------------------------------------------
class ResNet18(tf.keras.models.Model):
    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = tf.keras.layers.Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)

        return y

model = ResNet18([2, 2, 2, 2])
# ----------------------------------------------------------
# ----------------------------------------------------------


# ----------------------------------------------------------
# 编译模型
# ----------------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ----------------------------------------------------------
# 断点续练模型
# ----------------------------------------------------------
checkpoint_save_path = r"./checkpoint/cifar_ResNet18/ResNet18.ckpt"
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
                    epochs=10, 
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
weight_txt = os.path.join(csv_file, "cifar_ResNet18.txt")
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