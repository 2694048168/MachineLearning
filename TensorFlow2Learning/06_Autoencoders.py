#!/usr/bin/env python3
# coding: utf-8

# Pipeline for this case
# 1. Initial Imports
# 2. Loading and Processing the Data
# 3. Add noise to images
# 4. Building the Model
# 5. Compile and Train the Model
# 6. Denoising noise iamges

# 1. Initial Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Method of shielding output log information in tensorflow
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
tf.get_logger().setLevel('ERROR')

# 2. Loading and Processing the Data
# We don't need y_train and y_test
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
print("The max value of x_train is: {}".format(x_train[0].max()))
print("The min value of x_train is: {}".format(x_train[0].min()))
print(x_train.shape)
print(x_test.shape)

# normalization operation with the following lines:
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# from (60000, 28, 28) to (60000,28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_test.shape)

# 3. Add noise to images
noise_factor = 0.6
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# We also need to make sure that our array item values are within the range of 0 to 1. 
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0, clip_value_max=1)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0, clip_value_max=1)

# 4. Building the Model
class Denoise(tf.keras.Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([tf.keras.layers.Input(shape=(28, 28, 1)),
                                       tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same", strides=2),
                                       tf.keras.layers.Conv2D(8, (3,3), activation="relu", padding="same", strides=2)])
    self.decoder = tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation="relu", padding="same"),
                                        tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation="relu", padding="same"),
                                        tf.keras.layers.Conv2D(1, kernel_size=(3,3), activation="sigmoid", padding="same")])                                  

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)                                        
    return decoded

# Let’s create a model object with the following code:
autoencoder = Denoise()

# 5. Compile and Train the Model
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# 6. Denoising noise iamges
# Now that we trained our model, we can easily do denoising tasks. 
# For the simplicity of the prediction process, we use the test dataset. 
# But, feel free to process and try other images such as digits in the MNIST dataset.
encoded_imgs=autoencoder.encoder(x_test).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs).numpy()

# visualization the result
n = 10
plt.figure(figsize=(20,6))
for i in range(n):
  # display original + noise
  bx = plt.subplot(3, n, i + 1)
  plt.title("original + noise")
  plt.imshow(tf.squeeze(x_test_noisy[i]))
  plt.gray()
  bx.get_xaxis().set_visible(False)
  bx.get_yaxis().set_visible(False)

  # display reconstruction
  cx = plt.subplot(3, n, i + n + 1)
  plt.title("reconstructed")
  plt.imshow(tf.squeeze(decoded_imgs[i]))
  plt.gray()
  cx.get_xaxis().set_visible(False)
  cx.get_yaxis().set_visible(False)
  
  # display original
  ax = plt.subplot(3, n, i + 2*n + 1)
  plt.title("original")
  plt.imshow(tf.squeeze(x_test[i]))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
