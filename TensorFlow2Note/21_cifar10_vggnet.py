#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 21_cifar10_vggnet.py
@Function: TensroFlow2 API tf.keras 实现 AlexNet 对 CIFAR10 分类
@Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
@Linking: https://arxiv.org/abs/1409.1556
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
@Usage: $ python 21_cifar10_vggnet.py 
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
# 建立 VGGNet 模型
# 卷积核的个数逐渐增加, 以 2 的幂次进行增加;
# 因为越靠后,特征图的尺寸越小;
# 通过增加卷积核的个数来增加了特征图的深度, 从而保持了信息的承载能力
# ----------------------------------------------------------
class VGG16Net(tf.keras.models.Model):
    def __init__(self):
        super(VGG16Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层 1
        self.bn1 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act1 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.bn2 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act2 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop1 = tf.keras.layers.Dropout(0.2)  # dropout 层

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act3 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act4 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop2 = tf.keras.layers.Dropout(0.2)  # dropout 层

        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act5 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn6 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act6 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.act7 = tf.keras.layers.Activation('relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop3 = tf.keras.layers.Dropout(0.2)

        self.conv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn8 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act8 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv9 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn9 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act9 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv10 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn10 = tf.keras.layers.BatchNormalization()
        self.act10 = tf.keras.layers.Activation('relu')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop4 = tf.keras.layers.Dropout(0.2)

        self.conv11 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn11 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act11 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn12 = tf.keras.layers.BatchNormalization()  # BN 层 1
        self.act12 = tf.keras.layers.Activation('relu')  # 激活层 1
        self.conv13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.bn13 = tf.keras.layers.BatchNormalization()
        self.act13 = tf.keras.layers.Activation('relu')
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop5 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.drop6 = tf.keras.layers.Dropout(0.2)
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.drop7 = tf.keras.layers.Dropout(0.2)
        self.fc3 = tf.keras.layers.Dense(10, activation='softmax')
        # --------------------------------------------------------------------

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.act7(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.act8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.act9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.act10(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.act11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.act12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.act13(x)
        x = self.pool5(x)
        x = self.drop5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop6(x)
        x = self.fc2(x)
        x = self.drop7(x)
        y = self.fc3(x)
        
        return y
    # --------------------------------------------------------------------

# 实例化模型
model = VGG16Net()


# ----------------------------------------------------------
# 编译模型
# ----------------------------------------------------------
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# ----------------------------------------------------------
# 断点续练模型
# ----------------------------------------------------------
checkpoint_save_path = r"./checkpoint/cifar_VGG16Net/VGG16Net.ckpt"
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
weight_txt = os.path.join(csv_file, "cifar_VGG16Net.txt")
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