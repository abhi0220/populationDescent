import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy.special import softmax
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
from collections import namedtuple
from sklearn.cluster import KMeans
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
#import keras.backend as K
import copy
from copy import deepcopy
import tensorflow as tf


NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# FUNCTIONS FOR NN IMPLEMENTATION
def new_NN_individual():

	# # FM Model (small) 
	# model = tf.keras.Sequential([
	# tf.keras.layers.Flatten(input_shape=(28, 28)),
 #    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
 #    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
 #    tf.keras.layers.Dense(10)
	# ])

	# # FM Model (small --> get equal results with PD and .fit)
	# model = tf.keras.Sequential([
	# tf.keras.layers.Flatten(input_shape=(28, 28)),
 #    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.0001)), # good rate from hyperparameter search = 1e-4, 1e-5
 #    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.0001)),
 #    tf.keras.layers.Dense(10)
	# ])

	# # Keras Tutorial Model --> use for just overfitting rn
	# model = tf.keras.Sequential([
	# tf.keras.layers.Flatten(input_shape=(28, 28)),
	# tf.keras.layers.Dense(2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l=.0001)),
	# tf.keras.layers.Dense(10)
	# ])


	# model #3: for trying to avoid overfitting, hyperparameter vs PD
	model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(1024),
	tf.keras.layers.Dense(512),
	tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
    tf.keras.layers.Dense(10)
	])

	# eager_model = tf.function(model)

	# ## big keras model

	# model = tf.keras.Sequential()
	# model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=FM_input_shape))
	# model.add(layers.Activation('relu'))
	# model.add(layers.Conv2D(filters=96, kernel_size=(3,3), strides=2))
	# model.add(layers.Activation('relu'))

	# # model.add(layers.Conv2D(filters=192, kernel_size=(3,3)))
	# # model.add(layers.Activation('relu'))
	# # model.add(layers.Conv2D(filters=192, kernel_size=(3,3), strides=2))
	# # model.add(layers.Activation('relu'))

	# model.add(layers.Flatten())
	# model.add(layers.BatchNormalization())
	# model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)))
	# model.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=.001)))
	# model.add(layers.Dense(512))

	# model.add(layers.Activation('relu'))

	# model.add(layers.Dense(10, activation="softmax"))


	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # 1e-3
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))
	print(LR_constant, reg_constant)

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)
	print(type(NN_object))
	print(""), print(NN_object), print("")

	return NN_object