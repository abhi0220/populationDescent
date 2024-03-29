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

# Testing population descent
def new_pd_NN_individual(lr=1e-3):


	# model #6, no_reg - better, bigger CIFAR10 model
	# model_num = "6 no_reg CIFAR"
	# model = tf.keras.Sequential([
    # tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)),
    # # tf.keras.layers.BatchNormalization(),
    
    # # tf.keras.layers.Dropout(0.2),
    
    # tf.keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    
    # tf.keras.layers.MaxPooling2D((4, 4)),
    # # tf.keras.layers.Dropout(0.2),

    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(256, activation = "relu"),

    # tf.keras.layers.Dense(10, activation = "softmax")
    # ])


	# # model #6 with reg - CIFAR10 model
	model_num = "6 with_reg CIFAR"
	model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)),
    # tf.keras.layers.BatchNormalization(),
    
    # tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D((4, 4)),
    # tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=0.001)),

    tf.keras.layers.Dense(10, activation = "softmax")
    ])



	# optimizer = tf.keras.optimizers.legacy.Adam() # 1e-3 (for FMNIST, CIFAR)
	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr) # 1e-3 (for FMNIST)
	# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-2) # CIFAR (doesn't work well)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num


# 

# Testing Hyperparameter search
def new_hps_NN_individual():

	# regularization_amount = [0.001]
	# learning_rate = [1e-3]

	# regularization_amount = [0.001, 0.01, 0.1]
	# learning_rate = [0.001, 0.01, 0.1]
	
	# regularization_amount = [0.1]
	learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
	regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

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

			# # model #4 without regularization (for ESGD model comparison)
			# model_num = "4_no_reg; 25 models"
			# FM_input_shape = (28, 28, 1)
			# model = tf.keras.Sequential([
			# tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
			# tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
			# tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			# tf.keras.layers.Flatten(),
			# tf.keras.layers.Dense(1024),
			# tf.keras.layers.Activation('relu'),
			# tf.keras.layers.Dropout(0.5),
			# tf.keras.layers.Dense(10, activation='softmax')
			# ])


			# # model #4 with regularization (for ESGD model comparison)
			# model_num = "4; 25 models"
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


			# # model #5 for CIFAR10
			# model_num = "5 CIFAR no_reg"
			# cifar_input_shape = (32, 32, 3)
			
			# model = tf.keras.Sequential([
			# tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=cifar_input_shape),
			# tf.keras.layers.MaxPooling2D((2,2)),
			# tf.keras.layers.Dropout(0.2),
			# tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu'),
			# tf.keras.layers.MaxPooling2D((2,2)),
			# tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
			# tf.keras.layers.Dropout(0.5),

			# tf.keras.layers.Flatten(),
			# tf.keras.layers.Dense(1024, activation='relu'),
			# tf.keras.layers.Dense(10, activation='softmax')
			# ])


			# # model #6 - better, bigger CIFAR10 model
			# learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
			# regularization_amount = [0]

			# model_num = "6 CIFAR no_reg"
			# model = tf.keras.Sequential([
		 #    tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    # tf.keras.layers.Dropout(0.2),
		    
		 #    tf.keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    tf.keras.layers.MaxPooling2D((2, 2)),
		 #    tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu'),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    tf.keras.layers.MaxPooling2D((2, 2)),
		 #    tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    tf.keras.layers.MaxPooling2D((4, 4)),
		 #    # tf.keras.layers.Dropout(0.2),

		 #    tf.keras.layers.Flatten(),
		 #    tf.keras.layers.Dense(256, activation = "relu"),

		 #    tf.keras.layers.Dense(10, activation = "softmax")
		 #    ])


		 	# model #6 with reg, better, bigger CIFAR10 model
			# learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
			# regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

			model_num = "6 CIFAR with_reg"
			model = tf.keras.Sequential([
		    tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)),
		    # tf.keras.layers.BatchNormalization(),
		    
		    # tf.keras.layers.Dropout(0.2),
		    
		    tf.keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'),
		    # tf.keras.layers.BatchNormalization(),
		    
		    tf.keras.layers.MaxPooling2D((2, 2)),
		    tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu'),
		    # tf.keras.layers.BatchNormalization(),
		    
		    tf.keras.layers.MaxPooling2D((2, 2)),
		    tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
		    # tf.keras.layers.BatchNormalization(),
		    
		    tf.keras.layers.MaxPooling2D((4, 4)),
		    # tf.keras.layers.Dropout(0.2),

		    tf.keras.layers.Flatten(),
		    tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),

		    tf.keras.layers.Dense(10, activation = "softmax")
		    ])


		 #    # model #7 - bigger CIFAR
			# model_num = "7 CIFAR no_reg"
			# model = tf.keras.Sequential([
		 #    tf.keras.layers.Conv2D(32,  kernel_size = 3, kernel_initializer='he_normal', activation='relu', input_shape = (32, 32, 3)),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    # tf.keras.layers.Dropout(0.2),
		    
		 #    tf.keras.layers.Conv2D(64, kernel_size = 3, kernel_initializer='he_normal', strides=1, activation='relu'),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    tf.keras.layers.MaxPooling2D((2, 2)),
		 #    tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu'),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    tf.keras.layers.MaxPooling2D((2, 2)),
		 #    tf.keras.layers.Conv2D(64, kernel_size = 3,kernel_initializer='he_normal', activation='relu'),
		 #    # tf.keras.layers.BatchNormalization(),
		    
		 #    tf.keras.layers.MaxPooling2D((4, 4)),
		 #    # tf.keras.layers.Dropout(0.2),

		 #    tf.keras.layers.Flatten(),
		 #    tf.keras.layers.Dense(512, activation = "relu"),
		 #    tf.keras.layers.Dense(256, activation = "relu"),
		 #    tf.keras.layers.Dense(128, activation = "relu"),

		 #    # tf.keras.layers.Dropout(0.1),
		 #    tf.keras.layers.Dense(10, kernel_initializer='glorot_uniform', activation = "softmax")
		 #    ])





			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate[l])

			model.compile(optimizer=optimizer,
			         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			         metrics=['accuracy'])

			population.append(model)
			reg_list.append(regularization_amount[r])

	population = np.array(population)


	return population, reg_list, model_num



