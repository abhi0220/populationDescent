# use Python 3.9
# python3.9 -m venv env
# source new3.9/bin/activate
# pip3.9 install -r requirements.txt
# python3.9 -m pd_classes_parameters

# # # use this for sin wave with noise tests

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

#from pop_descent import pop_descent

from pd_paramters import pop_descent_classes

## FUNCTIONS FOR NON-NN IMPLEMENTATION
# OPTIMIZER
def create_optimizer(normalized_objective):
	def optimizer(pop):
		#convert np.arrays to tensors
		adam = tf.keras.optimizers.Adam(learning_rate = 5e-2)
		#computing gradient values
		with tf.GradientTape() as tape:
			tensors = tf.Variable(pop)
			fitnesses = 1 - normalized_objective(tensors)

		#convert tensors back to np.arrays, multiply learning rate
		adam.minimize(fitnesses, var_list = [tensors], tape = tape)

		return tensors.numpy(), 1 - fitnesses.numpy()
	return optimizer

# RANDOMIZER
def simple_randomizer(population, normalized_amount, iteration = None):
	#Adds random noise from Gaussian distribution to individual
	mu, sigma = 0, 0.8
	gNoise = (np.random.normal(mu, sigma))*normalized_amount
	return population + gNoise

# NEW POPULATION
def new_population(pop_size):
	population = []
	hist = []
	for i in range(pop_size):
		#coordinate = np.array([random.random()*20. - 10, random.random()*20. - 10])
		#coordinate = np.array((random.random()*20.-10))
		coordinate = np.array((random.random()*1.-1))
		#coordinate = np.array(-.2)
		#coordinate = np.array(0.888)
		population.append(coordinate)
	population = np.array(population)

	return population, hist

# normalized loss functions
@tf.function
def normalized_quadratic(pop): 
	tensor_values = tf.math.pow(pop, 2)
	return tf.math.subtract(tf.cast(1, tf.float64), tf.math.sigmoid(tensor_values))

@tf.function
def normalized_sin(pop):
	values = tf.math.sin(pop)/2.0 + 0.5
	return 1 - values

@tf.function
def normalized_complex_sin(pop):
	values = (tf.math.sin(pop) + tf.math.sin(2*pop) + tf.math.sin(23*pop))/5.51 + 0.5
	return 1 - values

@tf.function
def normalized_quartic(pop):
	tensor_values = (4*tf.math.pow(pop, 4))-(3*(tf.math.pow(pop, 2))+(0.01*pop))
	return tf.math.divide(tf.cast(1, tf.float64), tf.math.add(tf.cast(1, tf.float64), tensor_values))

def normalized_Ackley(pop):
	return -20.0 * tf.math.exp(-0.2 * tf.math.sqrt(0.5 * (pop[0]**2 + pop[1]**2))) - tf.math.exp(0.5 * (tf.math.cos(2 * pi * pop[0]) + tf.math.cos(2 * pi * pop[1]))) + tf.math.exp(1) + 20


def make_drawing_things(objective_function, number_of_replaced_individuals):
	def graph_recorder(population, hist):
		hist.append(max(objective_function(population).numpy()))
		allX = np.array(population)
		allY = objective_function(population).numpy()

		unreplacedX = np.array(population[-number_of_replaced_individuals:])
		unreplacedY = objective_function(unreplacedX).numpy()

		replacedX = np.array(population[0:number_of_replaced_individuals])
		replacedY = objective_function(replacedX).numpy()

		space = np.linspace(min(allX)-10, max(allX)+10, len(population)*20)
		ySpace = objective_function(space)

		plt.plot(space, ySpace, zorder = 0)
		plt.scatter(unreplacedX, unreplacedY, c='g', zorder = 10)
		plt.scatter(replacedX, replacedY, c='r', zorder = 100)
		plt.ylabel('red = new randomized individual')

		plt.show(block=False), plt.pause(0.5), plt.close()
		return

	def min_recorder(population, hist):
		#creates graph of minimum function value as individuals are optimized
		optimized_y_vals = list(objective_function(population).numpy())
		#min
		min_y_value, min_index = min(optimized_y_vals), optimized_y_vals.index(min(optimized_y_vals))
		minimum_x_val, minimum_y_val = population[min_index], min_y_value
		#max
		max_y_value, max_index = max(optimized_y_vals), optimized_y_vals.index(max(optimized_y_vals))
		maximum_x_val, maximum_y_val = population[max_index], max_y_value
		#return min
		print("" + "Final minimized value of the function: "), print(maximum_x_val, maximum_y_val), print("")
		plt.plot(hist), plt.xlabel('function value'), plt.show(block=False), plt.pause(1), plt.close()
		return

	return graph_recorder, min_recorder


# FUNCTIONS FOR NN IMPLEMENTATION

def new_NN_individual():

	# Sin_Model_Regularization_Layers
	model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(1,)),
	tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	tf.keras.layers.Dense(1)
	])

	model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
              metrics=['accuracy'])


	optimizer = tf.keras.optimizers.Adam()
	LR_constant = 10**(np.random.normal(-3, 2)) * 0
	reg_constant = 10**(np.random.normal(-3, 2)) * 0

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)
	return NN_object


def NN_optimizer_manual_loss(NN_object):

	batch_size = 50
	epochs = 25
	normalized_training_loss, normalized_validation_loss = [], []

	#indices = np.arange(FM_train_images.shape[0])
	indices = np.random.random_integers(59999, size = (batch_size*20, ))
	vIndices = np.random.random_integers(4999, size = (batch_size*10, ))

	

	# = np.array(list(range(50)))

	# FM dataset
	random_batch_FM_train_images, random_batch_FM_train_labels = FM_train_images[indices], FM_train_labels[indices]
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[vIndices], FM_validation_labels[vIndices]
	
	# C10 dataset
	#random_batch_C10_train_images, random_batch_C10_train_labels = C10_train_images[indices], C10_train_labels[indices]

	optimizer = NN_object.opt_obj # for NN usage (check if I can access it through nn directly --> .nn)
	# optimizer = tf.keras.optimizers.Adam() # for polynomial regression
	simple_lossfn = tf.keras.losses.MeanSquaredError()

	# lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	# print(optimizer.learning_rate)

	for e in range(epochs):

		sIndices = np.random.choice(100, 100, replace=False)
		random_batch_sin_train_x, random_batch_sin_train_labels = sin_x_train[sIndices], sin_y_train[sIndices]
		
		# NN_object.nn.fit(random_batch_sin_train_x, random_batch_sin_train_labels, epochs=1, verbose=1, batch_size = batch_size)


		with tf.GradientTape() as tape:

		# 	# make a prediction using the model and then calculate the loss
			# model_loss = lossfn(random_batch_FM_train_labels, NN_object.nn(random_batch_FM_train_images))

		# 	# validation_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))
		# 	#NN_object.nn.fit(sin_x_train, noisyY, epochs=5, verbose=1)

			# model_loss = simple_lossfn(py, var_polynomial(px, coefficients)) # for polynomial regression

			model_loss = simple_lossfn(random_batch_sin_train_labels, NN_object.nn(random_batch_sin_train_x)) # noisy sin wave
			regularization_loss = NN_object.nn.losses
			validation_loss = simple_lossfn(sin_y_val, NN_object.nn(sin_x_val))
			reg_loss = (regularization_loss[0] + regularization_loss[1])
			mreg_loss = reg_loss * NN_object.reg_constant

			print("mreg los: %s" % mreg_loss), print("")

			total_training_loss = tf.math.multiply(NN_object.LR_constant, tf.add(model_loss, mreg_loss))

			print("%s --> total training loss (GD loss + reg losses)" % total_training_loss)

		# 	#model_loss = lossfn(random_batch_C10_train_labels, NN_object.nn(random_batch_C10_train_images))
		# 	#validation_loss = lossfn(C10_validation_labels, NN_object.nn(C10_validation_images))

		# print(" %s --> unnormalized training loss: %s" % model_loss), print("")

		# grads = tape.gradient(model_loss, coefficients) # for poly regression
		# optimizer.apply_gradients(zip(grads, coefficients)) # for polynomial regression

		# # calculate the gradients using our tape and then update the model weights
		# grads = tape.gradient(model_loss, NN_object.nn.trainable_variables)
		grads = tape.gradient(total_training_loss, NN_object.nn.trainable_variables)
		optimizer.apply_gradients(zip(grads, NN_object.nn.trainable_variables))

		# print(grads)
		# print(NN_object.nn.trainable_variables[0])

	normalized_training_loss.append(1/(1+(model_loss)))
	normalized_training_loss = np.array(normalized_training_loss)

	normalized_validation_loss.append(1/(1+(validation_loss)))
	normalized_validation_loss = np.array(normalized_validation_loss)

	normalized_training_loss, normalized_validation_loss = normalized_training_loss, normalized_validation_loss

	# print(""), print("normalized training loss: %s" % normalized_training_loss)
	# print("normalized validation loss: %s" % normalized_validation_loss)

	#print(model_loss)

	return normalized_training_loss, normalized_validation_loss

def NN_randomizer_manual_loss(NN_object, normalized_amount):
	print(""), print("RANDOMIZING")

	# randomizing NN weights
	model_clone = tf.keras.models.clone_model(NN_object.nn)
	model_clone.set_weights(np.array(NN_object.nn.get_weights()))

	mu, sigma = 0, 1e-4
	gNoise = (np.random.normal(mu, sigma))*(normalized_amount)

	weights = np.array((NN_object.nn.get_weights()))
	randomized_weights = weights + gNoise
	model_clone.set_weights(randomized_weights)

	# randomizing regularization rate
	mu, sigma = 0, 0.3
	randomization = 2**(np.random.normal(mu, sigma) * normalized_amount)
	print("reg randomization: %s" % randomization)
	new_reg_constant = (NN_object.reg_constant) * randomization
	print("NN_object.reg_constant: %s" % NN_object.reg_constant)
	# print(normalized_amount)
	print("new_reg_constant: %s" % new_reg_constant)

	# randomizing learning_rates
	mu, sigma = 0, 0.3
	randomization = 2**(np.random.normal(mu, sigma)*normalized_amount)
	print("LR randomization: %s" % randomization)
	new_LR_constant = (NN_object.LR_constant) * randomization
	print("NN_object.LR_constant: %s" % NN_object.LR_constant)
	# print(normalized_amount)
	print("new_reg_constant: %s" % new_LR_constant)

	new_NN_Individual = NN_Individual(model_clone, NN_object.opt_obj, new_LR_constant, new_reg_constant) # without randoimzed LR

	return new_NN_Individual


def evaluator(NN_object, total_hist, batch_hist):
	regression_NN_compiler(NN_object.nn) # only if using manual loss optimizer/randomizer
	total_test_loss, batch_test_loss, test_acc = [], [], []
	batch_size = 64
	# indices = np.random.random_integers(9999, size = (batch_size*25, ))
	indices = np.arange(FM_test_images.shape[0])
	print(""), print(""), print("Evaluating models on test data after randomization")

	random_batch_FM_test_images = FM_test_images[indices]
	random_batch_FM_test_labels = FM_test_labels[indices]

	random_batch_C10_test_images = C10_test_images[indices]
	random_batch_C10_test_labels = C10_test_labels[indices]


	# ntest_loss = NN_object.nn.evaluate(sin_x_train, (np.sin(sin_x_train)), batch_size = 64, verbose=1) # testing sin+noise curve-fitting
	
	print(""), print("evaluating on sin wave (no noise)")
	all_test_loss = NN_object.nn.evaluate(sin_x_test, np.sin(sin_x_test), batch_size = 64, verbose=1) # comparing to sin wave (no noise)

	print(""), print("evaluating on test data (with noise)")
	partial_test_loss = NN_object.nn.evaluate(sin_x_test, sin_y_test, batch_size = 64, verbose=1) # comparing to sin test data (with noise)
	#ntest_loss = 0
	# NN_object.nn(test_sin_x_train[0])

	ntest_acc = 0
	# ntest_loss = NN_object.nn.evaluate(FM_train_images, FM_train_labels, batch_size = batch_size, verbose=1)
	#ntest_loss, ntest_acc = model.evaluate(random_batch_C10_test_images, random_batch_C10_test_labels, batch_size = batch_size, verbose=1)
	
	total_test_loss.append(all_test_loss)
	batch_test_loss.append(partial_test_loss)
	test_acc.append(ntest_acc)

	avg_total_loss = np.mean(total_test_loss)
	avg_batch_loss = np.mean(batch_test_loss)
	avg_acc = np.mean(test_acc)

	total_hist.append(avg_total_loss)
	print(""), print("avg_total_loss: %s" % avg_total_loss), print("avg_batch_loss: %s" % avg_batch_loss), print(""), print("")

	return avg_total_loss, avg_batch_loss, avg_acc

# External Evaluator
def Parameter_class_evaluator(population, total_hist, batch_hist):
	total_test_loss, batch_test_loss, all_acc = [], [], []

	# # for sin wave
	models = []
	for h in range(len(population)):
	# for h in range(1):
		m = population[h].nn
		plt.scatter(sin_x_train, sin_y_train, color = 'k')
		plt.scatter(sin_x_val, sin_y_val, color = 'b')
		plt.scatter(sin_x_test, sin_y_test, color = 'r')
		plt.plot(sin_x_data, np.sin(sin_x_data)) # sin function
		plt.plot(sin_x_data, m(sin_x_data)) # model predictions
		plt.title("WITHout CV; Figure %s" % (h+1))
		plt.xlabel("black = train;    blue = val;    red = test")
		plt.show(block=False), plt.pause(0), plt.close()
	# end of sin wave code

	# plt.scatter(sin_x_train, noisyY)
	# plt.plot(sin_x_train, np.sin(sin_x_train))
	# plt.plot(sin_x_train, population[0].nn(sin_x_train)) # plot from the 
	# plt.show(block=False), plt.pause(0), plt.close() # for noisy sin

	for i in range(len(population)):
		individual_total_loss, individual_batch_loss, individual_acc = evaluator(population[i], total_hist, batch_hist)

		total_test_loss.append(individual_total_loss)
		batch_test_loss.append(individual_batch_loss)
		all_acc.append(individual_acc)

	avg_total_loss = np.mean(total_test_loss)
	avg_batch_loss = np.mean(batch_test_loss)
	avg_acc = np.mean(all_acc)

	return avg_total_loss, avg_batch_loss, avg_acc

def classification_NN_compiler(model):
	model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	return

# def soft_acc(y_true, y_pred):
# 	return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def regression_NN_compiler(model):
	model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())
	return


# CLASSES
Individual = TypeVar('Individual')

@dataclass
class Parameters(Generic[Individual]):

	population: Callable[[int], np.array]
	randomizer: Callable[[np.array, float], np.array]
	optimizer: Callable[[np.array], np.array]

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
	individual_optimizer: Callable[[Individual], Individual]
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


	Parameters_object = Parameters(Parameter_new_population, Parameter_class_randomizer, Parameter_class_optimizer)
	return Parameters_object


def create_Parameters_NN_object(pop_size):

	population = []
	object = individual_to_params(pop_size, new_NN_individual, NN_randomizer_manual_loss, NN_optimizer_manual_loss)
	object.population = object.population(pop_size) # initiazling population

	return object



# curve fitting

#data
n = 5 # of data points
p = n # degree of polynomial

# roots = np.random.uniform(-2, 3, n) # Generates random roots for a 5th degree polynomial
# polyCoefficients = np.poly(roots) # Find coefficients of polynomial that has said roots

px = np.arange(1, 6) # X values
py = np.random.randint(0, 10, n)
# print(px)
# print(py)

# plt.scatter(px, py)
# plt.pause(5), plt.close()


# polynomial_y = np.polyval(polyCoefficients, original_polynomial_x) # Corresponding Y values passed through polynomial

coefficients = []

for z in range(p):
	coefficients.append(tf.Variable(tf.zeros([1, 1])))

# coefficients = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
# coefficients = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
# coefficients = np.array(coefficients)
# print(coefficients)


def var_polynomial(x, coefficients):
	pred_y = 0
	for z in range(p):
		pred_y+=coefficients[z]*(x**(z))
	return pred_y




# # # SIN wave with noise dataset (CV set is just regular sin wave, not split from same dataset)
# num = 50

# # training data
# sin_x_train = np.linspace(-3.14, 3.14, num=num).reshape([num, 1])
# y = np.sin(sin_x_train)

# mu, sigma = 0, 1
# noise = np.random.normal(mu, sigma, size=[num, 1])
# noisyY = noise + y
# noisyY = np.array(noisyY)

# # val Data
# sin_x_val = sin_x_train
# sin_y_val = y

# # Testing Data
# sin_x_test = np.linspace(-3, 3, num=num).reshape([num, 1])
# sin_y_test = np.sin(sin_x_test)

# # # graph noise + sin wave
# # plt.scatter(sin_x_train, noisyY)
# # plt.plot(sin_x_train, np.sin(sin_x_train))
# # plt.show(block=False), plt.pause(0), plt.close()


# # SIN wave with noise dataset (CV set is just regular sin wave, not split from same dataset)
num = 300

# full data
sin_x_data = np.linspace(-3.14, 3.14, num=num).reshape([num, 1])
sin_y_data = np.sin(sin_x_data)

# perturbing y data points
mu, sigma = 0, 1
noise = np.random.normal(mu, sigma, size=[num, 1])
noisyY = noise + sin_y_data
noisyY = np.array(noisyY)

indices = (np.arange(num))
np.random.shuffle(indices)

div = (num)//3

training_indices = indices[0:div]
val_indices = indices[div:(div*2)]
test_indices = indices[(div*2):(div*3)]

# training data
sin_x_train = sin_x_data[training_indices]
sin_y_train = noisyY[training_indices]

# val Data
sin_x_val = sin_x_data[val_indices]
sin_y_val = noisyY[val_indices]

# Testing Data
sin_x_test = sin_x_data[test_indices]
sin_y_test = noisyY[test_indices]
sin_test_data = (sin_x_train, sin_y_train)

# # graph noise + sin wave
# plt.scatter(sin_x_train, noisyY)
# plt.plot(sin_x_train, np.sin(sin_x_train))
# plt.show(block=False), plt.pause(0), plt.close()


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




# CIFAR-10 dataset
cifar_10 = tf.keras.datasets.cifar10
(C10_train_images, C10_train_labels), (C10_test_images, C10_test_labels) = cifar_10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# normalizing data
C10_train_images, C10_test_images = C10_train_images / 255.0, C10_test_images / 255.0


# SOME PARAMETERS FOR QUICK ACCESS
iterations = 1000
pop_size = 20
number_of_replaced_individuals = 8
normalized_objective = normalized_quartic


## MAIN RUNNING CODE

if __name__ == "__main__":

	optimizer = create_optimizer(normalized_objective)
	graph_recorder, min_recorder = make_drawing_things(normalized_objective, number_of_replaced_individuals)

	#Parameters_object = create_object(pop_size)
	Parameters_object = create_Parameters_NN_object(pop_size)

	#optimized_population, optimized_fitnesses = pop_descent(optimizer = optimizer, randomizer = simple_randomizer, new_population = new_population, pop_size = pop_size, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations, final_observer = min_recorder, recorder = graph_recorder, normalized_objective = normalized_complex_sin, normalized_randomness_strength = None)
	#optimized_population, optimized_fitnesses = pd_parameters(Parameters_object, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations, final_observer = None, recorder = None, normalized_objective = normalized_Ackley, normalized_randomness_strength = None)

	loss_data, acc_data, total_test_loss, batch_test_loss, total_test_acc = [], [], [], [], []


	for i in range(1):

		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

		#optimized_population, lfitnesses, vfitnesses = pop_descent(NN_optimizer, NN_randomizer, new_NN_population, pop_size = pop_size, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations, final_observer = None, recorder = evaluator, normalized_objective = None, normalized_randomness_strength = None)
		optimized_population, lfitnesses, vfitnesses = pop_descent_classes(Parameters_object, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations)

		lmean = statistics.mean(lfitnesses)
		loss_data.append(lmean)

		# evaluate from outside
		total_hist, batch_hist = [], []
		avg_total_loss, avg_batch_loss, avg_test_acc = Parameter_class_evaluator(optimized_population, total_hist, batch_hist)

		total_test_loss.append(avg_total_loss)
		batch_test_loss.append(avg_batch_loss)
		total_test_acc.append(avg_test_acc)

		#amean = statistics.mean(vfitnesses)
		#acc_data.append(amean)
	total_test_loss = np.array(total_test_loss)
	total_test_loss = 1/(1+(total_test_loss))
	batch_test_loss = np.array(batch_test_loss)
	batch_test_loss = 1/(1+(batch_test_loss))

	print(""), print("WITHout CV"), print(""), print(""), print("avg final normalized loss of population at end of iterations on training")
	print(loss_data)
	# print(acc_data)

	print(""), print("normalized_batch_losses (sin with no noise), normalized_test_losses (sin test data with noise)")
	print(total_test_loss), print(batch_test_loss)

