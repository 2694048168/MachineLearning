#!/usr/bin/env python3
# coding: utf-8

# Pipeline for this case
# 1. Initial Imports
# 2. Loading and Processing the Data
# 3. Building the GAN Model
#    - Generator Network
#    - Discriminator Network
#    - Configure GAN Network
#        - loss function
#        - Optimizer
#    - Set checkpoint
# 4. Train GAN Model
#    - Train Step
#    - Train Loop
#    - Image Generation Function
#    - Start Training
# 5.  Animate Generated Digits During the Training

# 1. Initial Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import(Dense,BatchNormalization,LeakyReLU,Reshape,Conv2DTranspose,Conv2D,Dropout,Flatten)
import os
import PIL
import glob # The glob module is used for Unix style pathname pattern expansion.
import imageio # The library that provides an easy interface to read and write a wide range of image data
import time
from IPython import display # A command shell for interactive computing in Python.

# 2. Loading and Processing the Data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# reshape and normalize(in range of -1 to 1)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Batch and shuffle the data
BUFFER_SIZE = 10000
BATCH_SIZE = 64
# ndarray convert tensor
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 3. Building the GAN Model
#    - Generator Network
def make_generator_model():
  model = tf.keras.Sequential()
  model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
  model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1),padding="same", use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2),padding="same", use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2),padding="same", use_bias=False, activation="tanh"))
  assert model.output_shape == (None, 28, 28, 1)
  
  return model

# declare our network with the following code:
generator = make_generator_model()

#    - Discriminator Network
def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same",input_shape=[28, 28, 1]))
  model.add(LeakyReLU())
  model.add(Dropout(0.3))
  model.add(Conv2D(128, (5, 5), strides=(2, 2),padding="same"))
  model.add(LeakyReLU())
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(1))
  
  return model

discriminator = make_discriminator_model()

#    - Configure GAN Network
#        - loss function
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

#        - Optimizer
generator_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

#    - Set checkpoint
# By using the os library, setting a path to save all the training steps with the following lines:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir, "ckpt")
# https://tensorflow.google.cn/api_docs/python/tf/train/Checkpoint
# Manages saving/restoring trackable values to disk.
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 4. Train GAN Model
EPOCHS = 60
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

#    - Train Step
# The following code with excessive comments are for the training step. 
# Please read the comments carefully

# tf.function annotation causes the function
# to be "compiled" as part of the training
@tf.function
def train_step(images):
  # 1 - Create a random noise to feed it into the model for the image generation
  noise = tf.random.normal([BATCH_SIZE, noise_dim])
  # 2 - Generate images and calculate loss values
  # GradientTape method records operations for automatic differentiation.
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)
    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images,training=True)
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    
  # 3 - Calculate gradients using loss values and model variables
  # "gradient" method computes the gradient using
  # operations recorded in context of this tape (gen_tape and disc_tape).
  # It accepts a target (e.g., gen_loss) variable and 
  # a source variable (e.g.,generator.trainable_variables)
  # target --> a list or nested structure of Tensors or Variables to be differentiated.
  # source --> a list or nested structure of Tensors or Variables.
  # target will be differentiated against elements in sources.
  # "gradient" method returns a list or nested structure of Tensors
  # (or IndexedSlices, or None), one for each element in sources.
  # Returned structure is the same as the structure of sources.
  gradients_of_generator = gen_tape.gradient(gen_loss,generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient( disc_loss,discriminator.trainable_variables)
  
  # 4 - Process Gradients and Run the Optimizer
  # "apply_gradients" method processes aggregated gradients.
  # ex: optimizer.apply_gradients(zip(grads, vars))
  """
  Example use of apply_gradients:
  grads = tape.gradient(loss, vars)
  grads = tf.distribute.get_replica_context().all_reduce('sum',grads)
  # Processing aggregated gradients.
  optimizer.apply_gradients(zip(grads, vars), experimental_aggregate_gradients=False)
  """
  generator_optimizer.apply_gradients(zip( gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip( gradients_of_discriminator, discriminator.trainable_variables))
  
#    - Train Loop
# for this script IPython.display need to modify!!!
def train(dataset, epochs):
  # A. For each epoch, do the following:
  for epoch in range(epochs):
    start = time.time()
    # 1 - For each batch of the epoch,
    for image_batch in dataset:
      # 1.a - run the custom "train_step" function
      # we just declared above
      train_step(image_batch)
    
    # 2 - Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,epoch + 1,seed)
    
    # 3 - Save the model every 5 epochs as
    # a checkpoint, which we will use later
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      
    # 4 - Print out the completed epoch no. and the time spent
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
  # B. Generate a final image after the training is completed
  display.clear_output(wait=True)
  generate_and_save_images(generator,epochs,seed)
  
#    - Image Generation Function
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  # 1 - Generate images
  predictions = model(test_input, training=False)
  
  # 2 - Plot the generated images
  fig = plt.figure(figsize=(4,4))
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5,cmap="gray")
    plt.axis('off')
    
  # 3 - Save the generated images
  plt.savefig('./GAN_images/image_at_epoch_{:04d}.png'.format(epoch))

#    - Start Training
train(train_dataset, EPOCHS)

# Now that we trained our model and saved our checkpoints, 
# we can restore the trained model with the following line:
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# 5.  Animate Generated Digits During the Training
# PIL is a library which may open different image file formats
def display_image(epoch_no):
  return PIL.Image.open( './GAN_images/image_at_epoch_{:04d}.png'.format(epoch_no))

anim_file = './GAN_images/dcgan.gif'
with imageio.get_writer(anim_file, mode="I") as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
