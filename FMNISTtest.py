# use Python 3.9
# python3.9 -m venv new3.9
# source new3.9/bin/activate
# pip3.9 install -r requirements.txt
# pip3.9 install -r requirements_m1.txt

import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
import tensorflow as tf

import time
start_time = time.time()

from populationDescent import populationDescent
from NN_models import new_NN_individual

# NN_Individual = namedtuple("NN_Individual", "nn opt_obj LR_constant reg_constant")
NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])
tf.config.run_functions_eagerly(True)


def NN_optimizer_manual_loss(NN_object):
	
	# classification_NN_compiler(NN_object.nn)
	batch_size = 64
	epochs = 1
	normalized_training_loss, normalized_validation_loss = [], []

	print(""), print(NN_object), print("")
	optimizer = NN_object.opt_obj
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	indices = np.random.choice(59999, size = (batch_size*21, ), replace=False)
	vIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

	# FM dataset
	random_batch_FM_train_images, random_batch_FM_train_labels = FM_train_images[indices], FM_train_labels[indices]
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[vIndices], FM_validation_labels[vIndices]

	model_loss = gradient_steps(lossfn, random_batch_FM_train_images, random_batch_FM_train_labels, epochs, NN_object)

	validation_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))
	tf.print("validation loss: %s" % validation_loss), print("")

	normalized_training_loss.append(2/(2+(model_loss)))
	normalized_training_loss = np.array(normalized_training_loss)

	normalized_validation_loss.append(2/(2+(validation_loss)))
	normalized_validation_loss = np.array(normalized_validation_loss)

	print(""), print("normalized training loss: %s" % normalized_training_loss)
	print("normalized validation loss: %s" % normalized_validation_loss)

	#print(model_loss)
	return normalized_training_loss, normalized_validation_loss

# function optimized to take gradient steps with tf variables
@tf.function
def gradient_steps(lossfn, training_set, labels, epochs, NN_object):

	for e in range(epochs):
		
		with tf.GradientTape() as tape:

			# make a prediction using the model and then calculate the loss
			model_loss = lossfn(labels, NN_object.nn(training_set))
		
			# use regularization constant
			regularization_loss = NN_object.nn.losses
			reg_loss = regularization_loss[0]
			# reg_loss = ((regularization_loss[0] + regularization_loss[1]))
			mreg_loss = reg_loss * NN_object.reg_constant
			# mreg_loss = reg_loss * 1

			# total_training_loss = tf.math.multiply(nn.LR_constant, model_loss) # LR randomization
			# total_training_loss = model_loss + mreg_loss # REG randomization
			total_training_loss = NN_object.LR_constant * (model_loss + mreg_loss) # LR + REG randomization

			tf.print("training loss: %s" % model_loss) ## remove this --> put nothing (put at recombination)
			# print("mreg los: %s" % mreg_loss), print("")
			# print("total loss: %s" % total_training_loss), print("")

	# 	# print(" %s --> unnormalized training loss: %s" % model_loss), print("")

	# 	# # calculate the gradients using our tape and then update the model weights
		grads = tape.gradient(model_loss, NN_object.nn.trainable_variables)
		# grads = gradient_steps(tape, model_loss, nn)
		# grads = tape.gradient(total_training_loss, nn.nn.trainable_variables) ## with LR randomization
		NN_object.opt_obj.apply_gradients(zip(grads, NN_object.nn.trainable_variables))

	return model_loss


def NN_randomizer_manual_loss(NN_object, normalized_amount):
	print(""), print("RANDOMIZING")
	# original: (0, 1e-3), (0, normalized_amount), (0, normalized amount)

	factor = 25

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

# unnormalized
def observer(NN_object, tIndices):
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[tIndices], FM_validation_labels[tIndices]

	print(type(NN_object))
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	test_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))

	ntest_loss = 1/(1+test_loss)

	return test_loss

def graph_history(history, trial, parameter_string, loss_data_string, best_training_model_string, best_test_model_loss_string):
	integers = [i for i in range(1, (len(history))+1)]
	x = [j * rr for j in integers]
	y = history

	plt.scatter(x, history, s=20)
	# plt.rcParams.update({'font.size': 10})
	# figure(figsize=(3, 2), dpi=80)

	plt.tight_layout()
	plt.title("PD trial #%s" % trial)
	plt.ylabel('unnormalized loss of best model')
	plt.xlabel('iterations')

	plt.xlabel("%s\n\n%s\n\n%s\n\n%s" % (parameter_string, loss_data_string, best_training_model_string, best_test_model_loss_string))

	# for i,j in zip(x,y):
	# 	plt.annotate(str(j),xy=(i,j))
	plt.text(x[(len(x))-1], y[(len(y))-1], y[(len(y))-1])
	plt.axhline(y = y[(len(y))-1])

	plt.tight_layout()
	plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
	plt.show(block=True), plt.pause(0.5), plt.close()


def evaluator(NN_object):
	# classification_NN_compiler(NN_object) # only if using manual loss optimizer/randomizer
	batch_size = 64

	np.random.seed(0)
	tIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
	random_batch_FM_test_images, random_batch_FM_test_labels = FM_test_images[tIndices], FM_test_labels[tIndices]
	
	print(""), print(""), print("Evaluating models on test data after randomization")

	# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	print(type(NN_object))
	test_loss = lossfn(random_batch_FM_test_labels, NN_object.nn(random_batch_FM_test_images))

	# test_loss, test_acc = NN_object.nn.evaluate(random_batch_FM_test_images, random_batch_FM_test_labels, batch_size = batch_size)

	ntest_loss = 1/(1+test_loss)
	print("unnormalized test loss: %s" % test_loss)
	print("normalized (1/1+loss) test loss: %s" % ntest_loss)

	return ntest_loss

# External Evaluator
def Parameter_class_evaluator(population):
	all_test_loss, all_acc = [], []

	for i in range(len(population)):
		individual_total_loss = evaluator(population[i])

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
	observer: Callable[[np.array], np.array] # check this for typing
	randomization: bool
	CV_selection: bool
	rr: int
	history: [np.array]

# updated np typing
	# population: Callable[[int], npt.NDArray[Individual]]
	# randomizer: Callable[[npt.NDArray[Individual], float], np.Array[Individual]]
	# optimizer: Callable[[npt.NDArray[Individual]], np.Array[Individual]]



def individual_to_params(
	pop_size: int,
	new_individual: Callable[[], Individual],
	individual_randomizer: Callable[[Individual, float], Individual],
	individual_optimizer: Callable[[Individual], Individual],
	observer: Callable[[Individual], float],
	randomization: bool,
	CV_selection: bool,
	rr: int, # randomization rate
	history: [float]
	) -> Parameters[Individual]:

	def Parameter_new_population(pop_size: int) -> np.array(Individual):
		population = np.zeros(pop_size, dtype=object)
		for i in range(pop_size):
			population[i]= new_individual()
		return population

	def Parameter_class_randomizer(population: np.array(Individual), normalized_amount: float) -> np.array(Individual):
		randomized_population = np.zeros(len(population), dtype=object)
		for i in range(len(population)):
			new_object = individual_randomizer(population[i], normalized_amount[i])
			randomized_population[i] = new_object

		return randomized_population

	def Parameter_class_optimizer(population: np.array(Individual)) -> np.array(Individual):
		lFitnesses, vFitnesses = [], []
		for i in range(len(population)):
			print(""), print("model #%s" % (i+1)), print("")
			normalized_training_loss, normalized_validation_loss = individual_optimizer(NN_Individual(*population[i]))
			lFitnesses.append(normalized_training_loss)
			vFitnesses.append(normalized_validation_loss)

		lFitnesses = np.array(lFitnesses)
		lFitnesses = lFitnesses.reshape([len(lFitnesses), ])

		vFitnesses = np.array(vFitnesses)
		vFitnesses = vFitnesses.reshape([len(vFitnesses), ])

		return lFitnesses, vFitnesses

	# (during optimization)
	def Parameter_class_observer(population, history):

		batch_size = 64
		tIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

		all_test_loss = []
		for i in range(len(population)):
			unnormalized_model_loss = observer(population[i], tIndices)
			all_test_loss.append(unnormalized_model_loss)

		avg_test_loss = np.mean(all_test_loss)
		best_test_model_loss = np.min(all_test_loss)

		history.append(best_test_model_loss) ## main action of observer (to graph optimization progress later)
		return

	Parameters_object = Parameters(Parameter_new_population, Parameter_class_randomizer, Parameter_class_optimizer, Parameter_class_observer, randomization, CV_selection, rr, history)
	return Parameters_object


def create_Parameters_NN_object(pop_size, randomization, CV_selection, rr):
	history = []

	object = individual_to_params(pop_size, new_NN_individual, NN_randomizer_manual_loss, NN_optimizer_manual_loss, observer, randomization=randomization, CV_selection=CV_selection, rr=rr, history=history)
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




trial = 1

# Parameters
iterations = 1000
pop_size = 5
number_of_replaced_individuals = 2
randomization = True
CV_selection = False
rr = 15 # leash for exploration

batch_size = 64
batches = 21


## MAIN RUNNING CODE
if __name__ == "__main__":

	#creating object to pass into pop descent
	Parameters_object = create_Parameters_NN_object(pop_size, randomization, CV_selection, rr)

	#creating lists to store data
	loss_data, acc_data, total_test_loss, batch_test_loss, total_test_acc = [], [], [], [], []

	# number of "trials"
	for i in range(1):

		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

		#RUNNING OPTIMIZATION
		optimized_population, lfitnesses, vfitnesses, history = populationDescent(Parameters_object, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations)

		#measuring how long optimization took
		time_lapsed = time.time() - start_time
		print(""), print(""), print("time:"), print("--- %s seconds ---" % time_lapsed), print(""), print("")


		best_model = np.max(lfitnesses)
		lmean = statistics.mean(lfitnesses)
		loss_data.append(lmean)

		# evaluate from outside
		total_hist, batch_hist = [], []
		avg_total_loss, best_test_model_loss = Parameter_class_evaluator(optimized_population)

	print(""), print("Title: PD vs Hyperparameter Search")
	parameter_string = "CV_sel: %s, randomize=%s, %s iterations, %s models, %s replaced, rr=%s" % (CV_selection, randomization, iterations, pop_size, number_of_replaced_individuals, rr)
	print(""), print(parameter_string)
	print(""), print("")
	loss_data_string = "avg normalized training loss of population on last epoch: %s" % loss_data
	print(loss_data_string)
	best_training_model_string = "normalized training loss of best model: %s" % best_model
	print(best_training_model_string)
	best_training_model_string_unnormalized = "unnormalized training loss of best model: %s" % ((1/best_model)-1)
	print(best_training_model_string_unnormalized)

	print("")
	# print("normalized average test loss: %s" % avg_total_loss)
	print("")
	best_test_model_loss_string = "normalized (1/1+loss) best model test loss: %s" % best_test_model_loss
	print(best_test_model_loss_string)
	best_test_model_string_unnormalized = "unnormalized test loss of best model: %s" % ((1/best_test_model_loss)-1)
	print(best_test_model_string_unnormalized)

	print("")
	print("time lapsed: %s" % time_lapsed)

	graph_history(history, trial, parameter_string, loss_data_string, best_training_model_string, best_test_model_loss_string)


# print("--- %s seconds ---" % (time.time() - start_time))
