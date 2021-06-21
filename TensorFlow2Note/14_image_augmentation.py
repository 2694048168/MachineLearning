#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 14_image_augmentation.py
@Function: TensroFlow2 API tf.keras 实现图像增强技术
            tf.keras.preprocessing.image.ImageDataGenerator()
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

# # ------------------------------------------------
# # 2. 数据增强，扩充数据集
# # ------------------------------------------------
# tf.keras.preprocessing.image.ImageDataGenerator(
#     featurewise_center=False, 
#     samplewise_center=False,
#     featurewise_std_normalization=False, 
#     samplewise_std_normalization=False,
#     zca_whitening=False, 
#     zca_epsilon=1e-06, 
#     rotation_range=0, 
#     width_shift_range=0.0,
#     height_shift_range=0.0, 
#     brightness_range=None, 
#     shear_range=0.0, 
#     zoom_range=0.0,
#     channel_shift_range=0.0, 
#     fill_mode='nearest', 
#     cval=0.0,
#     horizontal_flip=False, 
#     vertical_flip=False, 
#     rescale=None,
#     preprocessing_function=None, 
#     data_format=None, 
#     validation_split=0.0, 
#     dtype=None
# )
# # ------------------------------------------------
# image_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=所有数据将乘以该数值,
#     rotation_range=随机旋转角度范围,
#     width_shift_range=随机宽度偏移量,
#     height_shift_range=随机高度偏移量,
#     horizontal_flip=是否随机水平翻转,
#     zoom_range=随机缩放的范围[1-n, 1+n],
# )
# # transformer for train data
# image_augmentation.fit(x_train)
# # # 模型训练
# model.fit(image_augmentation.flow(x_train, y_train, batch_size=batch_size))
# # ------------------------------------------------


# ------------------------------------------------
# 数据集增强技术
# 显示原始图像和增强后的图像
# ------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

image_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=False,
    zoom_range=0.5
)
# image_gen_train.fit(x_train) 
# 这里要求 x_train 必须是四维的张量，所以需要对 x_train 进行 reshape 操作
image_augmentation.fit(x_train)
# ------------------------------------------------


# ------------------------------------------------
print("xtrain", x_train.shape)
x_train_subset1 = np.squeeze(x_train[:12])
print("xtrain_subset1", x_train_subset1.shape)

print("xtrain", x_train.shape)
x_train_subset2 = x_train[:12]  # 一次显示12张图片
print("xtrain_subset2",x_train_subset2.shape)


# ------------------------------------------------
fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')
# ------------------------------------------------
# 显示原始图片
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train_subset1[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()


# ------------------------------------------------
# 显示增强后的图片
fig = plt.figure(figsize=(20, 2))
for x_batch in image_augmentation.flow(x_train_subset2, batch_size=12, shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i + 1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break;
# ------------------------------------------------