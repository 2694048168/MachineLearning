#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 01_backpropagation.py
@Function: 反向传播和梯度下降优化方法
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ---------------------------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

tf.config.set_soft_device_placement(True)

# ---------------------------------------------------------------------
w = tf.Variable(tf.constant(5, dtype=tf.float32))
learning_rate = 0.2
# lr初始值：0.2 修改学习率 0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数 w
epochs = 40
loss_values = []
csv_file = r"./csv_file"
os.makedirs(csv_file, exist_ok=True)
weight_loss_csv = os.path.join(csv_file, "weight_loss.csv")
image_path = r"./images"
os.makedirs(image_path, exist_ok=True)


# ---------------------------------------------------------------------
def plot_loss(epochs, loss_values, image_path):
    plt.plot(epochs, loss_values, color="red")
    plt.legend(["Weight Gradient Decent"], loc="upper right")
    plt.title("Loss Function $(w + 1)^2$")
    plt.xlabel("epochs")
    plt.ylabel("loss value")
    plt.savefig(os.path.join(image_path, "weight_loss.png"), format='png', dpi=500)
    # plt.show()
    plt.close()


# ---------------------------------------------------------------------
for epoch in range(epochs):
    # AUTO Gradient compute
    with tf.GradientTape() as tape:
        loss_function = tf.square(w+1) 
    grads_w = tape.gradient(loss_function, w)

    w.assign_sub(learning_rate * grads_w)

    loss_values.append(loss_function.numpy())

    print(f"After {epoch} epoch, weight W is {w.numpy()}, loss is {loss_function.numpy():.6f}")

dataframe_loss = pd.DataFrame({"w_loss": loss_values})
dataframe_loss.to_csv(weight_loss_csv)
plot_loss(range(epochs), loss_values, image_path)


# # ---------------------------------------------------------------------
# # 学习率进行指数衰减
# # ---------------------------------------------------------------------
# def plot_lr_deacy(len_lr, lr_values):
#     plt.plot(len_lr, lr_values, color="red")
#     plt.legend(["Learning Rate Decent"], loc="upper right")
#     plt.title("Learning Rate Exp Deacy")
#     plt.xlabel("epochs")
#     plt.ylabel("learning rate value")
#     plt.show()
#     plt.close()

# w = tf.Variable(tf.constant(5, dtype=tf.float32))

# epochs = 40
# LR_BASE = 0.2  # 最初学习率
# LR_DECAY = 0.99  # 学习率衰减率
# LR_STEP = 1  # 喂入多少轮 BATCH_SIZE 后，更新一次学习率
# learning_rate_deacy = []

# # 指数衰减学习率 = 初始学习率 * 学习率衰减率 **(当前轮数 / 多少轮衰减一次)
# # 轮数：step or epoch

# for epoch in range(epochs):
#     lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
#     with tf.GradientTape() as tape:  # with 结构到 grads 框起了梯度的计算过程
#         loss = tf.square(w + 1)
#     grads = tape.gradient(loss, w)  # .gradient 函数告知谁对谁求导

#     w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
#     print(f"After {epoch} epoch, w is {w.numpy()}, loss is {loss:.6f}, lr is {lr}")

#     learning_rate_deacy.append(lr)

# plot_lr_deacy(range(len(learning_rate_deacy)), learning_rate_deacy)