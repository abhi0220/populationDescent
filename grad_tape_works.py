# use Python 3.9
# pip3.9 install -r requirements.txt
# python3.9 -m venv env
# source new3.9/bin/activate

import random
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import dax tasets, layers, models
#import keras.backend as K
import copy
from copy import deepcopy
import tensorflow as tf

#from pop_descent import pop_descent

from pd_paramters import pop_descent_classes

## FUNCTIONS FOR NON-NN IMPLEMENTATION
# OPTIMIZER

# FUNCTIONS FOR NN IMPLEMENTATION

def new_NN_individual():
# FM model
	model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,)),
    tf.keras.layers.Dense(700, activation='relu'),
    tf.keras.layers.Dense(700, activation='relu'),
    tf.keras.layers.Dense(1)
	])

	model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
              metrics=['accuracy'])


	optimizer = tf.keras.optimizers.Adam()
	NN_object = NN_Individual(model, optimizer)

	return NN_object


def train(NN_object):

	batch_size = 50
	epochs = 1000
	normalized_training_loss, normalized_validation_loss = [], []

	num = 50

	#sin_x_values = np.array(np.arange(-(np.pi*100), (np.pi*100), (np.pi)/4))
	sin_x_values = np.linspace(-10.0, 10, num=num).reshape([num, 1])

	y = np.sin(sin_x_values)


	noise = np.random.normal(loc=0.0, scale=1.0, size=[num, 1])

	noisyY = noise + y
	optimizer = NN_object.opt_obj
	lossfn = tf.keras.losses.MeanSquaredError()

	for e in range(epochs):

		sIndices = np.random.choice(batch_size, batch_size, replace=False)
		random_batch_sin_train_x, random_batch_sin_train_labels = sin_x_values[sIndices], noisyY[sIndices]
		print(random_batch_sin_train_labels.shape)
		print(random_batch_sin_train_x.shape)
		
		# NN_object.nn.fit(random_batch_sin_train_x, random_batch_sin_train_labels, epochs=1, verbose=1, batch_size = batch_size)


		with tf.GradientTape() as tape:

		# 	# make a prediction using the model and then calculate the loss
			model_loss = lossfn(random_batch_sin_train_labels, NN_object.nn(random_batch_sin_train_x))


		print(""), print("unnormalized: "), print(model_loss), print("")

		# grads = tape.gradient(model_loss, coefficients) # for poly regression
		# optimizer.apply_gradients(zip(grads, coefficients)) # for polynomial regression

		# # calculate the gradients using our tape and then update the model weights
		grads = tape.gradient(model_loss, NN_object.nn.trainable_variables)
		optimizer.apply_gradients(zip(grads, NN_object.nn.trainable_variables))

		# print(grads)
		# print(NN_object.nn.trainable_variables[0])

	# normalized_training_loss.append(1/(1+(model_loss)))
	# normalized_training_loss = np.array(normalized_training_loss)

	# # normalized_validation_loss.append(1/(1+(validation_loss)))
	# # normalized_validation_loss = np.array(normalized_validation_loss)
	# normalized_training_loss, normalized_validation_loss = normalized_training_loss, 0

	print(""), print("training"), print(normalized_training_loss)
	#print("validation"), print(validation_loss)

	#print(model_loss)

	return normalized_training_loss, normalized_validation_loss




# updated np typing
	# population: Callable[[int], npt.NDArray[Individual]]
	# randomizer: Callable[[npt.NDArray[Individual], float], np.Array[Individual]]
	# optimizer: Callable[[npt.NDArray[Individual]], np.Array[Individual]]

@dataclass
class NN_Individual:

	nn: models.Sequential()
	opt_obj: Adam()



# curve fitting




## MAIN RUNNING CODE

if __name__ == "__main__":
	train(new_NN_individual())
