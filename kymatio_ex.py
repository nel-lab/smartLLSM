#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:53:15 2021

@author: jimmytabet
"""

#%% imports
'''
conda create -n sm_kymatio tensorflow-estimator=2.0.0 tensorflow-gpu=2.0.0 spyder matplotlib scikit-learn pandas opencv
conda activate sm_kymatio
pip install kymatio
'''

import os
import tensorflow as tf
import kymatio
from kymatio.keras import Scattering2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

print('tf version:', tf.__version__)
print('kymatio version:', kymatio.__version__)
print('tf GPU?:', tf.test.is_gpu_available())

#%% FCN example
input_shape=(800,800)

model = tf.keras.models.Sequential([tf.keras.layers.Input(input_shape),
                                    Scattering2D(J=3),
                                    tf.keras.layers.Conv2D(16, 3, data_format='channels_first'),
                                    # tf.keras.layers.Flatten(),
                                    # tf.keras.layers.Dense(10)
                                    ])

model.summary()

#%% mnist example
"""
Classification of MNIST with scattering
=======================================
Here we demonstrate a simple application of scattering on the MNIST dataset.
We use 10000 images to train a linear classifier. Features are normalized by
batch normalization.
"""

###############################################################################
# Preliminaries
# -------------
#
# Since we're using TensorFlow and Keras to train the model, import the
# relevant modules.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

###############################################################################
# Finally, we import the `Scattering2D` class from the `kymatio.keras`
# package.

from kymatio.keras import Scattering2D

###############################################################################
# Training and testing the model
# ------------------------------
#
# First, we load in the data and normalize it.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255., x_test / 255.

###############################################################################
# We then create a Keras model using the scattering transform followed by a
# dense layer and a softmax activation.

inputs = Input(shape=(28, 28))
x = Scattering2D(J=3, L=8)(inputs)
x = Flatten()(x)
x_out = Dense(10, activation='softmax')(x)
model = Model(inputs, x_out)

###############################################################################
# Display the created model.

model.summary()

###############################################################################
# Once the model is created, we couple it with an Adam optimizer and a
# cross-entropy loss function.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###############################################################################
# We then train the model using `model.fit` on a subset of the MNIST data.

model.fit(x_train[:10000], y_train[:10000], epochs=15,
          batch_size=64, validation_split=0.2)

###############################################################################
# Finally, we evaluate the model on the held-out test data.

model.evaluate(x_test, y_test)
