import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy.special import softmax
import numpy as np

# Typing
# import typing
# from typing import TypeVar, Generic
# from collections.abc import Callable

from tqdm import tqdm
from collections import namedtuple
# from sklearn.cluster import KMeans
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#import keras.backend as K
import copy
from copy import deepcopy
import tensorflow as tf


NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# Testing population descent
def new_pd_NN_individual_FMNIST():

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

	# # model #2: Keras Tutorial Model --> use for just overfitting rn
	# model_num = 2
	# model = tf.keras.Sequential([
	# tf.keras.layers.Flatten(input_shape=(28, 28)),
	# tf.keras.layers.Dense(2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	# tf.keras.layers.Dense(10)
	# ])


	# # model #3: for trying to avoid overfitting, hyperparameter vs PD
	# model = tf.keras.Sequential([
	# tf.keras.layers.Flatten(input_shape=(28, 28)),
	# tf.keras.layers.Dense(1024),
	# tf.keras.layers.Dense(512),
	# tf.keras.layers.Dense(256),
 #    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
 #    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
 #    tf.keras.layers.Dense(10)
	# ])

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




	# model #4 for FMNIST without regularization (for ESGD model comparison)
	model_num = "4_no_reg"
	FM_input_shape = (28, 28, 1)
	
	model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
	tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024),
	tf.keras.layers.Activation('relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10, activation='softmax')
	])

	# # model #4 with regularization
	# model_num = "4_with_reg"
	# FM_input_shape = (28, 28, 1)
	# model = tf.keras.Sequential([
	# tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
	# tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
	# tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


	# tf.keras.layers.Flatten(),
	# tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	# tf.keras.layers.Activation('relu'),
	# tf.keras.layers.Dropout(0.5),
	# tf.keras.layers.Dense(10, activation='softmax')
	# ])

	


	print(model.summary())


	# optimizer = tf.keras.optimizers.legacy.Adam() # 1e-3 (for FMNIST, CIFAR)
	# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)
	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)
	# LR_constant = 10**(np.random.normal(-4, 2))
	LR_constant = 1
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num




# Testing Hyperparameter search
def new_hps_NN_individual_FMNIST():

	# regularization_amount = [0.001]
	# learning_rate = [1e-3]

	# regularization_amount = [0.001, 0.01, 0.1]
	# learning_rate = [0.001, 0.01, 0.1]

	# regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
	
	regularization_amount = [0.1]
	learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

	# regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
	# learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5]


	population = []
	reg_list = []

	for r in range(len(regularization_amount)):
		for l in range(len(learning_rate)):


			# # model #2: Keras Tutorial Model --> use for just overfitting rn
			# model_num = 2
			# model = tf.keras.Sequential([
			# tf.keras.layers.Flatten(input_shape=(28, 28)),
			# tf.keras.layers.Dense(2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
			# tf.keras.layers.Dense(10)
			# ])

			# # model #3: for trying to avoid overfitting, hyperparameter vs PD
			# model = tf.keras.Sequential([
			# tf.keras.layers.Flatten(input_shape=(28, 28)),
			# tf.keras.layers.Dense(1024),
			# tf.keras.layers.Dense(512),
			# tf.keras.layers.Dense(256),
		 #    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		 #    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		 #    tf.keras.layers.Dense(10)
			# ])
			# model_num = 3

			# model #4 without regularization (for ESGD model comparison)
			model_num = "4_no_reg; 5 models"
			FM_input_shape = (28, 28, 1)
			model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(1024),
			tf.keras.layers.Activation('relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
			])


			# # model #4 with regularization (for ESGD model comparison)
			# model_num = "4_with_reg; 25 models"
			# FM_input_shape = (28, 28, 1)
			# model = tf.keras.Sequential([
			# tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
			# tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
			# tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			# tf.keras.layers.Flatten(),
			# tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
			# tf.keras.layers.Activation('relu'),
			# tf.keras.layers.Dropout(0.5),
			# tf.keras.layers.Dense(10, activation='softmax')
			# ])


			# # model #4 with regularization (for ESGD model comparison) without regularization
			# model_num = "4_with_reg; 25 models"
			# FM_input_shape = (28, 28, 1)
			# model = tf.keras.Sequential([
			# tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
			# tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
			# tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			# tf.keras.layers.Flatten(),
			# tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
			# tf.keras.layers.Activation('relu'),
			# # tf.keras.layers.Dropout(0.5),
			# tf.keras.layers.Dense(10, activation='softmax')
			# ])

			# # model 8, big from online: https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f
			# model_num = "8"
			# model = tf.keras.Sequential()
			# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last', input_shape=(28,28,1)))
			# # model.add(BatchNormalization())
			# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
			# # model.add(BatchNormalization())
			# model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
			# # model.add(Dropout(0.25))

			# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
			# # model.add(BatchNormalization())
			# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
			# # model.add(BatchNormalization())
			# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
			# # model.add(Dropout(0.25))

			# model.add(Flatten())
			# model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])))
			# # model.add(Dense(512, activation='relu'))
			# # model.add(BatchNormalization())
			# # model.add(Dropout(0.25))
			# model.add(Dense(1024, activation='relu'))
			# # model.add(BatchNormalization())
			# # model.add(Dropout(0.5))
			# model.add(Dense(10, activation='softmax'))

			# print(model.summary())

			# # model #7 (custom big) with regularization
			# model_num = "4_with_reg"
			# FM_input_shape = (28, 28, 1)
			# model = tf.keras.Sequential([
			# tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),

			# tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),
			# tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			# tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),
			# tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			# tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),
			# # tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			# tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),
			# # tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			# # tf.keras.layers.Conv2D(filters=2048, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),
			# # tf.keras.layers.Conv2D(filters=4096, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),
			# # tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			# tf.keras.layers.Flatten(),
			# # tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),
			# tf.keras.layers.Dense(1024),

			# tf.keras.layers.Dense(512),
			# tf.keras.layers.Dense(256),
			# tf.keras.layers.Dense(128),
			# tf.keras.layers.Dense(64),

			# tf.keras.layers.Activation('relu'),
			# # tf.keras.layers.Dropout(0.5),
			# tf.keras.layers.Dense(10, activation='softmax')
			# ])

			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate[l])

			model.compile(optimizer=optimizer,
			         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			         metrics=['accuracy'])

			population.append(model)
			reg_list.append(regularization_amount[r])

	population = np.array(population)


	return population, reg_list, model_num



