#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 22_cifar10_inception.py
@Function: TensroFlow2 API tf.keras 实现 InceptionNet 对 CIFAR10 分类
@Paper: Going Deeper with Convolutions
@Linking: https://arxiv.org/abs/1409.4842
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 22_cifar10_inception.py 
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
# 建立 InceptionNet 模型
# 卷积核的个数逐渐增加, 以 2 的幂次进行增加;
# 因为越靠后,特征图的尺寸越小;
# 通过增加卷积核的个数来增加了特征图的深度, 从而保持了信息的承载能力
# ----------------------------------------------------------
class ConvBNRelu(tf.keras.models.Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False) 
        # 在 training=False 时，BN通过整个训练集计算均值、方差去做批归一化;
        # training=True 时，通过当前 batch 的均值、方差去做批归一化。
        # 推理时 training=False 效果好

        return x

# ----------------------------------------------------------
# Inception 结构快: 在同一层网络内使用不同尺寸的卷积核, 提升模型感知力
# 多个尺度提取特征 1*1; 3*3; pooling;
# Inception 结构快的最后输出是将各个尺度提取的特征图按照深度方向做拼接
# 调整 batch_size , 使得 GPU 负载在 80%-90%
# ----------------------------------------------------------
class InceptionBlk(tf.keras.models.Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = tf.keras.layers.MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel; 按照深度方向做拼接
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)

        return x

# ----------------------------------------------------------
class Inception10(tf.keras.models.Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = Inception10(num_blocks=2, num_classes=10)
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
checkpoint_save_path = r"./checkpoint/cifar_Inception/Inception10.ckpt"
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
weight_txt = os.path.join(csv_file, "cifar_Inception.txt")
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