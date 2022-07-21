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

from pd_paramters import pop_descent_classes

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


	model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(1024),
	tf.keras.layers.Dense(512),
	tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
    tf.keras.layers.Dense(10)
	])

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
	return NN_object


def NN_optimizer_manual_loss(NN_object):

	# classification_NN_compiler(NN_object.nn)
	batch_size = 64
	epochs = 1
	normalized_training_loss, normalized_validation_loss = [], []

	optimizer = NN_object.opt_obj
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	indices = np.random.choice(59999, size = (batch_size*21, ), replace=False)
	vIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

	# FM dataset
	random_batch_FM_train_images, random_batch_FM_train_labels = FM_train_images[indices], FM_train_labels[indices]
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[vIndices], FM_validation_labels[vIndices]

	# NN_object.nn.fit(random_batch_FM_train_images, random_batch_FM_train_labels, validation_data = (random_batch_FM_validation_images, random_batch_FM_validation_labels), epochs=epochs, verbose=1, batch_size = batch_size)

	for e in range(epochs):
		
		with tf.GradientTape() as tape:

			# make a prediction using the model and then calculate the loss
			model_loss = lossfn(random_batch_FM_train_labels, NN_object.nn(random_batch_FM_train_images))
		
			# use regularization constant
			regularization_loss = NN_object.nn.losses
			reg_loss = regularization_loss[0]
			# reg_loss = ((regularization_loss[0] + regularization_loss[1]))
			mreg_loss = reg_loss * NN_object.reg_constant
			# mreg_loss = reg_loss * 1

			# total_training_loss = tf.math.multiply(NN_object.LR_constant, model_loss) # LR randomization
			# total_training_loss = model_loss + mreg_loss # REG randomization
			total_training_loss = NN_object.LR_constant * (model_loss + mreg_loss) # LR + REG randomization

			tf.print("training loss: %s" % model_loss)
			# print("mreg los: %s" % mreg_loss), print("")
			# print("total loss: %s" % total_training_loss), print("")

			validation_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))
			tf.print("validation loss: %s" % validation_loss)
			print("")
	# 	# print(" %s --> unnormalized training loss: %s" % model_loss), print("")

	# 	# # calculate the gradients using our tape and then update the model weights
		grads = tape.gradient(model_loss, NN_object.nn.trainable_variables)
		# grads = tape.gradient(total_training_loss, NN_object.nn.trainable_variables) ## with LR randomization
		optimizer.apply_gradients(zip(grads, NN_object.nn.trainable_variables))

	normalized_training_loss.append(2/(2+(model_loss)))
	normalized_training_loss = np.array(normalized_training_loss)

	normalized_validation_loss.append(2/(2+(validation_loss)))
	normalized_validation_loss = np.array(normalized_validation_loss)

	print(""), print("normalized training loss: %s" % normalized_training_loss)
	print("normalized validation loss: %s" % normalized_validation_loss)

	#print(model_loss)
	return normalized_training_loss, normalized_validation_loss

def NN_randomizer_manual_loss(NN_object, normalized_amount):
	print(""), print("RANDOMIZING")
	# original: (0, 1e-3), (0, normalized_amount), (0, normalized amount)

	factor = 100

	# randomizing NN weights
	model_clone = tf.keras.models.clone_model(NN_object.nn)
	model_clone.set_weights(np.array(NN_object.nn.get_weights()))

	mu, sigma = 0, (1e-2) #1e-4 for sin
	gNoise = (np.random.normal(mu, sigma))*(normalized_amount)

	weights = np.array((NN_object.nn.get_weights()))
	randomized_weights = weights + gNoise
	model_clone.set_weights(randomized_weights)

	# randomizing regularization rate
	mu, sigma = 0, (normalized_amount*factor) # 0.7, 1 #10 # 0.3
	print(mu, sigma)
	print("")
	randomization = 2**(np.random.normal(mu, sigma))
	new_reg_constant = (NN_object.reg_constant) * randomization

	print("reg randomization: %s" % randomization)
	print("%s NN_object.reg_constant" % NN_object.reg_constant)
	# print(normalized_amount)
	print("%s new_reg_constant" % new_reg_constant), print("")

	# randomizing learning_rates
	mu, sigma = 0, (normalized_amount*factor) # 0.7, 1 #10 # 0.3
	randomization = 2**(np.random.normal(mu, sigma))
	new_LR_constant = (NN_object.LR_constant) * randomization

	print("LR randomization: %s" % randomization)
	print("%s NN_object.LR_constant" % NN_object.LR_constant)
	print(normalized_amount)
	print("%s new_lr_constant" % new_LR_constant)
	print(""), print("factor=%s" % factor)

	new_NN_Individual = NN_Individual(model_clone, NN_object.opt_obj, new_LR_constant, new_reg_constant) # without randoimzed LR

	return new_NN_Individual


def evaluator(NN_object, total_hist, batch_hist):
	# classification_NN_compiler(NN_object) # only if using manual loss optimizer/randomizer
	batch_size = 64

	np.random.seed(0)
	tIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
	random_batch_FM_test_images, random_batch_FM_test_labels = FM_test_images[tIndices], FM_test_labels[tIndices]
	
	print(""), print(""), print("Evaluating models on test data after randomization")

	# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	test_loss = lossfn(random_batch_FM_test_labels, NN_object.nn(random_batch_FM_test_images))

	# test_loss, test_acc = NN_object.nn.evaluate(random_batch_FM_test_images, random_batch_FM_test_labels, batch_size = batch_size)

	ntest_loss = 1/(1+test_loss)
	print("unnormalized test loss: %s" % test_loss)
	print("normalized (1/1+loss) test loss: %s" % ntest_loss)

	return ntest_loss

def classification_NN_compiler(NN_object):
	NN_object.nn.compile(optimizer=NN_object.opt_obj,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	return

# External Evaluator
def Parameter_class_evaluator(population, total_hist, batch_hist):
	all_test_loss, batch_test_loss, all_acc = [], [], []

	for i in range(len(population)):
		individual_total_loss = evaluator(population[i], total_hist, batch_hist)

		all_test_loss.append(individual_total_loss)

	avg_total_test_loss = np.mean(all_test_loss)
	best_test_model_loss = np.max(all_test_loss)

	return avg_total_test_loss, best_test_model_loss

# CLASSES
Individual = TypeVar('Individual')

@dataclass
class Parameters(Generic[Individual]):

	population: Callable[[int], np.array]
	randomizer: Callable[[np.array, float], np.array]
	optimizer: Callable[[np.array], np.array]
	randomization: bool
	CV_selection: bool
	rr: int
	# observer: 

# updated np typing
	# population: Callable[[int], npt.NDArray[Individual]]
	# randomizer: Callable[[npt.NDArray[Individual], float], np.Array[Individual]]
	# optimizer: Callable[[npt.NDArray[Individual]], np.Array[Individual]]

@dataclass
class NN_Individual:

	nn: models.Sequential()
	opt_obj: Adam()
	LR_constant: np.cfloat
	reg_constant: np.cfloat


def individual_to_params(
	pop_size: int,
	new_individual: Callable[[], Individual],
	individual_randomizer: Callable[[Individual, float], Individual],
	individual_optimizer: Callable[[Individual], Individual],
	randomization: bool,
	CV_selection: bool,
	rr: int
	) -> Parameters[Individual]:

	def Parameter_new_population(pop_size: int) -> np.array(Individual):
		population = []
		for i in range(pop_size):
			individual = new_individual()
			population.append(individual)
		population = np.array(population)

		return population

	def Parameter_class_randomizer(population: np.array(Individual), normalized_amount: float) -> np.array(Individual):
		randomized_population = []
		for i in range(len(population)):
			new_object = individual_randomizer(population[i], normalized_amount[i])
			randomized_population.append(new_object)
		randomized_population = np.array(randomized_population)

		return randomized_population

	def Parameter_class_optimizer(population: np.array(Individual)) -> np.array(Individual):
		lFitnesses, vFitnesses = [], []
		for i in range(len(population)):
			print(""), print("model #%s" % (i+1)), print("")
			normalized_training_loss, normalized_validation_loss = individual_optimizer(population[i])
			lFitnesses.append(normalized_training_loss)
			vFitnesses.append(normalized_validation_loss)

		lFitnesses = np.array(lFitnesses)
		lFitnesses = lFitnesses.reshape([len(lFitnesses), ])

		vFitnesses = np.array(vFitnesses)
		vFitnesses = vFitnesses.reshape([len(vFitnesses), ])

		return lFitnesses, vFitnesses


	Parameters_object = Parameters(Parameter_new_population, Parameter_class_randomizer, Parameter_class_optimizer, randomization, CV_selection, rr)
	return Parameters_object


def create_Parameters_NN_object(pop_size, randomization, CV_selection, rr):

	object = individual_to_params(pop_size, new_NN_individual, NN_randomizer_manual_loss, NN_optimizer_manual_loss, randomization, CV_selection, rr)
	object.population = object.population(pop_size) # initiazling population

	return object


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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Parameters
iterations = 2000
pop_size = 8
number_of_replaced_individuals = 4
randomization = True
CV_selection = True
rr = 15 # leash for exploration

batch_size = 64
batches = 21


## MAIN RUNNING CODE
if __name__ == "__main__":

	#Parameters_object = create_object(pop_size)
	Parameters_object = create_Parameters_NN_object(pop_size, randomization, CV_selection, rr)

	#optimized_population, optimized_fitnesses = pop_descent(optimizer = optimizer, randomizer = simple_randomizer, new_population = new_population, pop_size = pop_size, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations, final_observer = min_recorder, recorder = graph_recorder, normalized_objective = normalized_complex_sin, normalized_randomness_strength = None)
	#optimized_population, optimized_fitnesses = pd_parameters(Parameters_object, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations, final_observer = None, recorder = None, normalized_objective = normalized_Ackley, normalized_randomness_strength = None)

	loss_data, acc_data, total_test_loss, batch_test_loss, total_test_acc = [], [], [], [], []

	for i in range(1):

		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

		#optimized_population, lfitnesses, vfitnesses = pop_descent(NN_optimizer, NN_randomizer, new_NN_population, pop_size = pop_size, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations, final_observer = None, recorder = evaluator, normalized_objective = None, normalized_randomness_strength = None)
		optimized_population, lfitnesses, vfitnesses = pop_descent_classes(Parameters_object, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations)

		best_model = np.max(lfitnesses)

		lmean = statistics.mean(lfitnesses)
		loss_data.append(lmean)

		# evaluate from outside
		total_hist, batch_hist = [], []
		avg_total_loss, best_test_model_loss = Parameter_class_evaluator(optimized_population, total_hist, batch_hist)

	print("Title: PD vs Hyperparameter Search")
	print(""), print("CV_selection: %s, randomization=%s, %s iterations, %s models, %s individuals replaced, rr=%s" % (CV_selection, randomization, iterations, pop_size, number_of_replaced_individuals, rr))
	print(""), print("")
	print("avg normalized training loss of population on last epoch: %s" % loss_data)
	print("normalized training loss of best model: %s" % best_model)

	print("")
	# print("normalized average test loss: %s" % avg_total_loss)
	print("")
	print("normalized (1/1+loss) best model test loss: %s" % best_test_model_loss)
	print("")


