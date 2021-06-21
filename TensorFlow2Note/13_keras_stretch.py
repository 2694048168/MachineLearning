#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 13_keras_stretch.py
@Function: TensroFlow2 API tf.keras 实现功能的延展
            1. 自己制作数据集
            2. 数据集增强
            3. 断点继续训练, 存取模型
            4. 模型参数提取, 存入文本
            5. 可视化 TensorBoard
            6. 进行测试和使用模型
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
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


# # ------------------------------------------------
# # 1. 自己制作数据集，解决本领域应用
# # ------------------------------------------------
# # 训练数据文件夹：mnist_train_jpg_60000
# # 训练数据对应标签文件：mnist_train_jpg_60000.txt: 第一列是图像文件名；第二列是图像对应的标签
# # 测试数据文件夹：mnist_test_jpg_60000
# # 测试数据对应标签文件：mnist_test_jpg_60000.txt：第一列是图像文件名；第二列是图像对应的标签
# # ------------------------------------------------
# train_path = './mnist_image_label/mnist_train_jpg_60000/'
# train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'
# x_train_savepath = './mnist_image_label/mnist_x_train.npy'
# y_train_savepath = './mnist_image_label/mnist_y_train.npy'

# test_path = './mnist_image_label/mnist_test_jpg_10000/'
# test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'
# x_test_savepath = './mnist_image_label/mnist_x_test.npy'
# y_test_savepath = './mnist_image_label/mnist_y_test.npy'
# # ------------------------------------------------
# def generateds(path, txt):
#     # f = open(txt, 'r')  # 以只读形式打开txt文件
#     # contents = f.readlines()  # 读取文件中所有行
#     # f.close()  # 关闭 txt文件
#     with open(txt, "r") as f:
#         contents = f.readlines()
# 
#     x, y_ = [], []  # 建立空列表
#     for content in contents:  # 逐行取出
#         # value = content.split(sep="")
#         value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
#         img_path = path + value[0]  # 拼出图片路径和文件名
#         img = Image.open(img_path)  # 读入图片
#         img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
#         img = img / 255.  # 数据归一化 （实现预处理）
#         x.append(img)  # 归一化后的数据，贴到列表x
#         y_.append(value[1])  # 标签贴到列表y_
#         print('loading : ' + content)  # 打印状态提示

#     x = np.array(x)  # 变为np.array格式
#     y_ = np.array(y_)  # 变为np.array格式
#     y_ = y_.astype(np.int64)  # 变为64位整型
# 
#     return x, y_  # 返回输入特征x，返回标签y_
# # ------------------------------------------------
# if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
#         x_test_savepath) and os.path.exists(y_test_savepath):
#     print('-------------Load Datasets-----------------')
#     x_train_save = np.load(x_train_savepath)
#     y_train = np.load(y_train_savepath)
#     x_test_save = np.load(x_test_savepath)
#     y_test = np.load(y_test_savepath)
#     x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
#     x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
# else:
#     print('-------------Generate Datasets-----------------')
#     x_train, y_train = generateds(train_path, train_txt)
#     x_test, y_test = generateds(test_path, test_txt)

#     print('-------------Save Datasets-----------------')
#     x_train_save = np.reshape(x_train, (len(x_train), -1))
#     x_test_save = np.reshape(x_test, (len(x_test), -1))
#     np.save(x_train_savepath, x_train_save)
#     np.save(y_train_savepath, y_train)
#     np.save(x_test_savepath, x_test_save)
#     np.save(y_test_savepath, y_test)
# # ------------------------------------------------


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


# # ------------------------------------------------
# # 3. 断点继续训练，存取模型
# # ------------------------------------------------
# # 读取模型：load_weights(路径文件名)
# checkpoint_save_path = r"./checkpoint/mnist.ckpt"
# os.makedirs(checkpoint_save_path, exists=True)
# if os.path.exists(checkpoint_save_path + ".index"):
#     print("------------ Load the Model ------------")
#     model.load_weights(checkpoint_save_path)
# # ------------------------------------------------
# # 保存模型
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=路径文件名,
#     save_weights_only=True/False,
#     save_best_only=True/False,
# )
# history = model.fit(callbacks=[cp_callback])
# # ------------------------------------------------


# # ------------------------------------------------
# # 4. 参数提取，将参数存入文本
# # ------------------------------------------------
# # 提取可训练参数
# # model.trainable_variables 返回模型中可训练的参数
# # 设置 print 输出格式
# # np.set_printoptions(threshold=超过多少省略显示)
# np.set_printoptions(threshold=np.inf)
# print(model.trainable_variables)
# with open("./weights.txt", "w") as f:
#     for var in model.trainable_variables:
#         f.write(str(var.name) + "\n")
#         f.write(str(var.shape) + "\n")
#         f.write(str(var.numpy()) + "\n")
# # ------------------------------------------------


# # ------------------------------------------------
# # 5. acc/loss 可视化，查看训练效果
# # ------------------------------------------------
# # 显示训练集和验证集的 acc 和 loss 曲线
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# plt.show()
# plt.close()
# # ------------------------------------------------


# ------------------------------------------------
# 神经网络进行功能延展
# ------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# ------------------------------------------------
# image_gen_train.fit(x_train) 
# 这里要求 x_train 必须是四维的张量，所以需要对 x_train 进行 reshape 操作
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

image_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
# transformer for train data
# image_augmentation.fit(x_train)
# ------------------------------------------------


# ------------------------------------------------
# 参数都是可以以 字符串形式给出；或者使用函数形式给出
# ------------------------------------------------
# 使用类继承的形式编写自定义的网络模型
# 前向计算
# 后向传播
# ------------------------------------------------
class MNISTModel(tf.keras.models.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.conv1 = tf.keras.layers.Dense(units=128, activation="relu", use_bias=True)
        self.conv2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.conv1(x)
        y = self.conv2(x)

        return y

# 实例化
model = MNISTModel()
# ------------------------------------------------


# ------------------------------------------------
# from_logits=True 询问是否是原始输出(没有经过概率分布的输出)
# 输出经过概率分布 (softmax)，False
# 输出没有经过概率分布 (softmax)，True
# ------------------------------------------------
# metrics 可选
# accuracy：y_hat and y_pred 都是数值，如 y_hat=[1] y_pred=[1]
# categorical_accuracy: y_hat and y_pred 都是独热编码格式 (概率分布)
# sparse_categorical_accuracy: y_hat 是数值， y_pred 是独热编码格式 (概率分布)
# ------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ------------------------------------------------
# checkpoint_save_path = r"./checkpoint/mnist.ckpt"
checkpoint_save_path = r"./checkpoint/mnist_model/mnist.ckpt"
if os.path.exists(checkpoint_save_path + ".index"):
    print("------------ Load the Model ------------")
    model.load_weights(checkpoint_save_path)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         )
# ------------------------------------------------


# ------------------------------------------------
# validation_freq 多少次 epoch 进行测试一次
# 经过数据增强
# ------------------------------------------------
# history = model.fit(image_augmentation.flow(x_train, y_train, batch_size=32), 
#           epochs=5, 
#           validation_data=(x_test, y_test),
#           validation_freq=1,
#           callbacks=[checkpoint_callback],
#           verbose=2)

history = model.fit(x_train, 
                    y_train, 
                    batch_size=32, 
                    epochs=5, 
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[checkpoint_callback],
                    verbose=2)
# ------------------------------------------------
model.summary()


# ------------------------------------------------
# print(model.trainable_variables)
csv_file = r"./csv_file"
os.makedirs(csv_file, exist_ok=True)
weight_txt = os.path.join(csv_file, "weights.txt")
with open(weight_txt, "w") as file:
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')

# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()


# ------------------------------------------------
# 显示训练集和验证集的 acc 和 loss 曲线
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
# ------------------------------------------------