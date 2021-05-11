#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# # checking supporting GPU for tensorflow
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

# # The first option is to turn on memory growth by calling
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# Pipeline for this case
# 1. TensorFlow Imports for Dataset Downloading
# 2. Preparing the Dataset for model
# 3. Building the Recurrent Neural Network
# 4. Compile model
# 5. Fit model
# 6. Evaluating model
# 7. Testing model
# 8. Saving model for loading

# 1. TensorFlow Imports for Dataset Downloading
# Dataset is a dictionary containing train, test, and unlabeled datasets
# Info contains relevant information about the dataset
dataset, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

# Understanding the Bag-of-Word Concept: Text Encoding and Decoding
# Using info can load the encoder which converts text to bag of words
encoder = info.features['text'].encoder

# 2. Preparing the Dataset for model
# We can easily split our dataset dictionary with the relevant keys
train_dataset, test_dataset = dataset['train'], dataset['test']
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)

# 3. Building the Recurrent Neural Network
model = tf.keras.models.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
                                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                                    tf.keras.layers.Dense(64, activation="relu"),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(1)])

# model.summary()                                    
# tf.keras.utils.plot_model(model, show_shapes=True)

# 4. Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])

# 5. Fit model
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)

# 6. Evaluating model
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# visualzation the loss for training
# We can also use our history object to plot the performance measures over time with the following code:
def plot_graphs(history, metric):
  plt.figure(figsize=(6, 6), dpi=120)
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  # plt.show()
  # https://www.osgeo.cn/matplotlib/api/_as_gen/matplotlib.pyplot.savefig.html
  plt.savefig("./iamge/RNN_loss", dpi=120, format=png)

plot_graphs(history, 'accuracy')

# 7. Testing model
# making New Predictions
# The following code is our custom padding function:
def review_padding(encoded_review, padding_size):
  zeros = [0] * (padding_size - len(encoded_review))
  encoded_review.extend(zeros)
  return encoded_review

# encoder function that would encode and process our review to feed into our trained model
def review_encoder(review):
  encoded_review = review_padding(encoder.encode( review ),64)
  encoded_review = tf.cast( encoded_review, tf.float32)
  return tf.expand_dims( encoded_review, 0)

fight_club_review = 'It has some cliched moments, even for its time, but FIGHT CLUB is an awesome film. \
                     I have watched it about 100 times in the past 20 years. \
                     It never gets old. It is hard to discuss this film without giving things away but suffice it to say,\
                     it is a great thriller with some intriguing twists.'

# predicting for new review
model.predict(review_encoder(fight_club_review))

# 8. Saving model for loading
# This will save the full model with its variables, weights, and biases.
model.save('/saved_model/sentiment_analysis')

# Also save the encoder for later use
encoder.save_to_file('/saved_model/sa_vocab')

# Load the Trained Model and Make Predictions
# you can shut dowm this kernel, and then starting the followingcell.
loaded = tf.keras.models.load_model("/saved_model/sentiment_analysis/")
vocab_path = '/saved_model/sa_vocab'
encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_path)

# predicting for new review
loaded.predict(review_encoder(rev))
