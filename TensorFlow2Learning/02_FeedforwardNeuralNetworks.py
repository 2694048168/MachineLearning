#!/usr/bin/env python
# coding: utf-8

# pipeline for this case
# 1. Initial Installs and Imports libraries
# 2. Downloading the Auto MPG Data
# 3. Data Preparation
#     - Attribute Information
#     - DataFrame Creation
#     - Dropping Null Values
#     - Handling Categorical Variables (Dummy Variable)
#     - Splitting Auto MPG for Training and Testing
#     - normalizer(x) function can be used for train, test, and new observation sets.
# Feature scaling with the mean
# and std. dev. values in train_stats
# 4. Model Building and Training
#     - Tensorflow Imports
#     - Model build with Sequential API
#     - Model Configuration
#     - Early Stop Configuration
#     - Fitting the Model and Saving the Callback Histories
# 5. Evaluating the Results
# 6. Making Predictions with a New Observation

# 1. Initial Installs and Imports libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
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

# 2. Downloading the Auto MPG Data
# tf.keras.utils.get_file(fname="auto-mpg", origin=url, cache_dir="./")

# 3. Data Preparation
#     - Attribute Information
column_names = ['mpg', 'cylinders', 'displacement', 'HP', 'weight', 'acceleration', 'modelyear', 'origin']

#     - DataFrame Creation
mpg_df = pd.read_csv("./dataset/auto-mpg.data", sep=" ", comment="\t", names=column_names, na_values=None, skipinitialspace=True)

#     - Dropping Null Values
mpg_df = mpg_df.dropna()
mpg_df = mpg_df.reset_index(drop=True)

#     - Handling Categorical Variables (Dummy Variable)
def one_hot_origin_encoder(mpg_df):
  mpg_df_copy = mpg_df.copy()
  mpg_df_copy["EU"] = mpg_df_copy["origin"].map({1:0, 2:1, 3:0})
  mpg_df_copy["Jpan"] = mpg_df_copy["origin"].map({1:0, 2:0, 3:1})
  mpg_df_copy = mpg_df_copy.drop("origin", axis=1)
  return mpg_df_copy

mpg_df_clean = one_hot_origin_encoder(mpg_df)

#     - Splitting Auto MPG for Training and Testing
train_set = mpg_df_clean.sample(frac=0.8, random_state=0)
train_X = train_set.drop("mpg", axis=1)
train_y = train_set["mpg"] 

test_set = mpg_df_clean.drop(train_set.index)
test_X = test_set.drop("mpg", axis=1)
test_y = test_set["mpg"] 

#     - normalizer(x) function can be used for train, test, and new observation sets.
train_stats = train_X.describe().transpose()

def normalizer(x):
  return (x - train_stats["mean"]) / train_stats["std"]

train_x_scaled = normalizer(train_X).astype(np.float32)
test_x_scaled = normalizer(test_X).astype(np.float32)

# 4. Model Building and Training
#     - Tensorflow Imports
#     - Model build with Sequential API
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=[train_x_scaled.shape[1]]))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))

#     - Model Configuration
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mae", metrics=["mae"])

#     - Early Stop Configuration
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

#     - Fitting the Model and Saving the Callback Histories
history = model.fit(x=train_x_scaled, y=train_y, epochs=1000, validation_split=0.2, verbose=0, callbacks=[early_stop])

# 5. Evaluating the Results
# visualization of training loop
plt.figure(figsize=(8,6), dpi=120)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('MAE(mpg)')
plt.legend()
plt.show()

# test metrics
loss, mae = model.evaluate(test_x_scaled, test_y, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# 6. Making Predictions with a New Observation
# What is the MPG of a car with the following info:
new_car = pd.DataFrame([[8, #cylinders
                         307.0, #displacement
                         130.0, #HP
                         5504.0, #weight
                         12.0, #acceleration
                         70, #modelyear
                         1 #origin
                         ]], columns=column_names[1:])

new_car = normalizer(one_hot_origin_encoder(new_car))
new_car_mpg = model.predict(new_car).flatten()
print('The predicted miles per gallon value for this car is:',new_car_mpg)
