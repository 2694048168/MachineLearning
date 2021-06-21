#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 19_cifar10_lenet5.py
@Function: TensroFlow2 API tf.keras 实现 LeNet5 对 CIFAR10 分类
@Linking: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 19_cifar10_lenet5.py 
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
# 建立 LeNet5 模型
# ----------------------------------------------------------
class LeNet5(tf.keras.models.Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=6, 
                                            kernel_size=(5, 5),
                                            strides=(1, 1),
                                            activation="sigmoid")
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=16, 
                                            kernel_size=(5, 5),
                                            strides=(1, 1),
                                            activation="sigmoid")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=120, activation="sigmoid")
        self.fc2 = tf.keras.layers.Dense(units=84, activation="sigmoid")
        self.fc3 = tf.keras.layers.Dense(units=10, activation="softmax")
    # --------------------------------------------------------------------

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)

        return y
    # --------------------------------------------------------------------

# 实例化模型
model = LeNet5()

# ----------------------------------------------------------
# 编译模型
# ----------------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ----------------------------------------------------------
# 断点续练模型
# ----------------------------------------------------------
checkpoint_save_path = r"./checkpoint/cifar_LeNet5/LeNet5.ckpt"
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
                    epochs=150, 
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
weight_txt = os.path.join(csv_file, "cifar_LeNet5.txt")
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