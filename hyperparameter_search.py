# use Python 3.9
# python3.9 -m venv env
# source new3.9/bin/activate
# pip3.9 install -r requirements.txt
# python3.9 -m pd_classes_parameters

import random
import math
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
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


regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

# 500000 gradient steps

population = []
reg_list = []

for r in range(len(regularization_amount)):
	for l in range(len(learning_rate)):

		# smallest model to try to force model.fits to get stuck
		# model = tf.keras.Sequential([
		# tf.keras.layers.Flatten(input_shape=(28, 28)),
		# tf.keras.layers.Dense(2, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		# tf.keras.layers.Dense(10)
		# ])

		# model = tf.keras.Sequential([
		# tf.keras.layers.Flatten(input_shape=(28, 28)),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(10)
		# ])

		# model = tf.keras.Sequential([
		# tf.keras.layers.Flatten(input_shape=(28, 28)),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
		#    tf.keras.layers.Dense(10)
		# ])

		model = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(1024),
		tf.keras.layers.Dense(512),
		tf.keras.layers.Dense(256),
	    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
	    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
	    tf.keras.layers.Dense(10)
		])

		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate[l])
		model.compile(optimizer=optimizer,
		         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		         metrics=['accuracy'])

		population.append(model)
		reg_list.append(regularization_amount[r])

population = np.array(population)



# Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(FM_train_images, FM_train_labels), (FM_test_images, FM_test_labels) = fashion_mnist.load_data()

sample_shape = FM_train_images[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
FM_input_shape = (img_width, img_height, 1)

# Reshape data 
FM_train_images = FM_train_images.reshape(len(FM_train_images), FM_input_shape[0], FM_input_shape[1], FM_input_shape[2])
FM_test_images  = FM_test_images.reshape(len(FM_test_images), FM_input_shape[0], FM_input_shape[1], FM_input_shape[2])

# normalizing data
FM_train_images, FM_test_images = FM_train_images / 255.0, FM_test_images / 255.0

# FM_validation_images, FM_validation_labels = FM_train_images[50000:59999], FM_train_labels[50000:59999]
# FM_train_images, FM_train_labels = FM_train_images[0:50000], FM_train_labels[0:50000]

FM_validation_images, FM_validation_labels = FM_test_images[0:5000], FM_test_labels[0:5000]
FM_test_images, FM_test_labels = FM_test_images[5000:], FM_test_labels[5000:]
print(len(FM_test_images))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


iterations = 100
epochs = 1
batch_size = 64

for i in tqdm(range(iterations)):

	indices = np.random.choice(59999, size = (batch_size*20, ), replace=False)
	vIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

	random_batch_FM_train_images, random_batch_FM_train_labels = FM_train_images[indices], FM_train_labels[indices]
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[vIndices], FM_validation_labels[vIndices]


	for j in range(len(population)):

		print("model %s" % (j+1))
		population[j].fit(random_batch_FM_train_images, random_batch_FM_train_labels, validation_data = (random_batch_FM_validation_images, random_batch_FM_validation_labels), epochs=epochs, verbose=1, batch_size = batch_size)

		print("regularization_amount: %s" % reg_list[j])
		print("learning rate: %s" % population[j].optimizer.learning_rate)
		print("")


# Evaluating on test data

# things to set seed for evaluation

np.random.seed(0)
tIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
random_batch_FM_test_images, random_batch_FM_test_labels = FM_test_images[tIndices], FM_test_labels[tIndices]

evaluation_losses, evaluation_accuracies = [], []

for h in range(len(population)):
	print("model %s" % (h+1))
	test_loss, test_acc = population[h].evaluate(random_batch_FM_test_images, random_batch_FM_test_labels, batch_size = batch_size)

	ntest_loss = 1/(1+test_loss)
	ntest_loss = np.array(ntest_loss)

	evaluation_losses.append(ntest_loss)
	evaluation_accuracies.append(test_acc)


best_test_model_loss = np.max(evaluation_losses)
best_index = evaluation_losses.index(best_test_model_loss)

best_lr = (population[best_index]).optimizer.learning_rate
best_reg_amount = reg_list[best_index]

evaluation_losses = np.array(evaluation_losses)
test_loss_data = statistics.mean(evaluation_losses)
test_acc_data = statistics.mean(evaluation_accuracies)




print("")
print("model #%s" % (best_index+1))
print("avg final normalized loss of population at end of iterations on training %s" % test_loss_data)
print("normalized test loss of best model: %s" % best_test_model_loss)
print("best LR: %s" % best_lr)
print("best reg amount: %s" % best_reg_amount), print("")





