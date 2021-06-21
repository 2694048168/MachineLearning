#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 16_fashion_train.py
@Function: TensroFlow2 API tf.keras 实现 FASHION 数据集分类
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
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion.load_data()
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
image_augmentation.fit(x_train)
# ------------------------------------------------


# ------------------------------------------------
# 参数都是可以以 字符串形式给出；或者使用函数形式给出
# ------------------------------------------------
# 使用类继承的形式编写自定义的网络模型
# 前向计算
# 后向传播
# ------------------------------------------------
class FASHIONModel(tf.keras.models.Model):
    def __init__(self):
        super(FASHIONModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.conv1 = tf.keras.layers.Dense(units=128, activation="relu", use_bias=True)
        self.conv2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.conv1(x)
        y = self.conv2(x)

        return y

# 实例化
model = FASHIONModel()
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
checkpoint_save_path = r"./checkpoint/fashion_model/mnist.ckpt"
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
history = model.fit(image_augmentation.flow(x_train, y_train, batch_size=32), 
          epochs=50, 
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
weight_txt = os.path.join(csv_file, "weights_fashion.txt")
with open(weight_txt, "w") as file:
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')


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