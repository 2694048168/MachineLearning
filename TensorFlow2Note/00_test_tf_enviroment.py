#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 00_test_tf_enviroment.py
@Function: 测试 tensroflow2 环境是否可以正常使用
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

# ---------------------------------------------------------------------
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.python.client import device_lib


tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# ---------------------------------------------------------------------
python_version = sys.version
print("---------------------------------")
print(f"Python Version: {python_version}")

tensorflow_version = tf.__version__
print("---------------------------------")
print(f"TensorFlow Version: {tensorflow_version}")

if physical_devices:
    for gpu in physical_devices:
        print("---------------------------------")
        print(f"The GPU available {gpu}")

print("---------------------------------")
print(device_lib.list_local_devices())


# ---------------------------------------------------------------------
X = tf.constant([1.0, 2.0], name="x")
Y = tf.constant([1.0, 2.0], name="y")
result = tf.add(X, Y, name="add")
print("---------------------------------")
print(result)