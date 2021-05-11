#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# checking supporting GPU for tensorflow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# Pipeline for this case
# 1. download mnist dataset
# 2. data processing for model
# 3. build CNN model
# 4. configure or compile model
# 5. train and fit model
# 6. evaluating model
# 7. saving model for using


# 1. download mnist dataset
# tensorflow dataset is useful to TensorFlow 
(x_train,y_train),(x_test,y_test)=tfds.as_numpy(tfds.load('mnist',split=['train', 'test'], 
                                                          batch_size=-1, #all data in single batch
                                                          as_supervised=True, #only input and label
                                                          shuffle_files=True #shuffle data to randomize
                                                         ))

# 2. data processing for model
# reshapeing and normalizing image data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("The shape of train set: {}".format(x_train.shape))
print("The shape of test set: {}".format(x_test.shape))
print("The samples for training: {}".format(x_train.shape[0]))
print("The samples for testing: {}".format(x_test.shape[0]))

# 3. build CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# 4. configure or compile model
# optimizer, loss function and metrices
model.compile(optimizer="adam", loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

# 5. train and fit model
model.fit(x=x_train, y=y_train, epochs=10)

# 6. evaluating model
model.evaluate(x=x_test, y=y_test)

# 7. predicting or testing model
image_index = 42
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print("Our CNN model predicts that the digit in this iamge is: {}".format(pred.argmax()))
# visualization the ditit image
plt.figure(figsize=(6,6))
plt.imshow(x_test[image_index].reshape(28, 28), cmap="Greys")
plt.show()

# 8. saving model for using
model.save("./saved_model/mnist_CNN")
