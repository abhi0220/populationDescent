import random
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import statistics
import tensorflow as tf

def pop_descent(optimizer, normalized_objective, new_population, randomizer, observer = None, pop_size = 100, iterations = 20):

	# optimizer: np.array(individuals) -> np.array(individual), np.array(floats(0-1))
	# new_population: () -> np.array(individuals)
	# randomizer: np.array(individuals) -> np.array(individuals)
	# observer: np.array(individuals) --> ()
	# pop_size: int (number of individuals)

	replaced_individuals = 30
	normalized_randomness_strength = 0.1

	# creating population of individuals - floats (coordinates)
	population = np.array(new_population(pop_size))
	hist = [] #list of history

	#MAIN LOOP
	for i in tqdm(range(0,iterations), desc = "Running Randomizer"):

		#calling OPTIMIZER, getting normalized fitness values
		population, fitnesses = optimizer(population, normalized_objective) #best = close to 1

		#sorting fitnesses/population from worst to best
		sorted_ind = np.argsort(fitnesses)
		fitnesses = fitnesses[sorted_ind]
		population = population[sorted_ind]

		#calling SIMPLE RANDOMIZER
		#population[0:replaced_individuals] = randomizer(population[-replaced_individuals:], normalized_randomness_strength)

		#calling WEIGHTED RANDOMIZER
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = fitnesses, k = replaced_individuals)))
		chosen_population = population[chosen_indices]
		randomizer_strength = 1 - (fitnesses[chosen_indices])
		
		population[0:replaced_individuals] = randomizer(chosen_population, randomizer_strength)

		#graph
		if(observer):
			observer(population, hist)

	min_recorder(population, hist)
	return population, fitnesses


# Optimizer
def optimizer(pop, normalized_objective = None):
	#convert np.arrays to tensors
	adam = tf.keras.optimizers.Adam(learning_rate = 1e-2)

	#computing gradient values
	with tf.GradientTape() as tape:
		tensors = tf.Variable(pop)
		fitnesses = 1 - normalized_objective(tensors)

	#convert tensors back to np.arrays, multiply learning rate
	adam.minimize(fitnesses, var_list = [tensors], tape = tape)

	return tensors.numpy(), 1 - fitnesses.numpy()


@tf.function
def normalized_quadratic(pop): 
	tensor_values = tf.math.pow(pop, 2)
	return tf.math.subtract(tf.cast(1, tf.float64), tf.math.sigmoid(tensor_values))

# def normalized_quadratic_np(pop):
# 	return normalized_quadratic(pop).numpy()

@tf.function
def normalized_sin(pop):
	values = tf.math.sin(pop)/2.0 + 0.5
	return 1 - values

# @tf.function
# def normalized_quartic(pop):
# 	tensor_values = (4*tf.math.pow(pop, 4))-(3*(tf.math.pow(pop, 2))+(0.01*pop))
# 	return tf.math.subtract(tf.cast(1, tf.float64), tf.math.sigmoid(tensor_values))

@tf.function
def normalized_quartic(pop):
	tensor_values = (4*tf.math.pow(pop, 4))-(3*(tf.math.pow(pop, 2))+(0.01*pop))
	return tf.math.divide(tf.cast(1, tf.float64), tf.math.add(tf.cast(1, tf.float64), tensor_values))


def new_population(pop_size):
	population = []
	for i in range(pop_size):
		coordinate = np.array((random.random()*1.-0.7))
		population.append(coordinate)
	return population

def simple_randomizer(population, normalized_amount):
	#Adds random noise from Gaussian distribution to individual
	mu, sigma = 0, 0.1
	gNoise = (np.random.normal(mu, sigma))*normalized_amount
	return population + gNoise


def make_drawing_things(objective_function):
	def recorder(population, hist):
		hist.append(max(objective_function(population).numpy()))
		x = np.array(population)
		y = objective_function(x).numpy()

		space = np.linspace(min(x)-10, max(x)+10, len(population)*20)
		ySpace = objective_function(space)

		plt.plot(space, ySpace), plt.scatter(x, y)
		plt.show(block=False), plt.pause(0.0005), plt.close()
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
		plt.plot(hist), plt.ylabel('function value'), plt.show(block=False), plt.pause(1), plt.close()
		return
	return recorder, min_recorder

recorder, min_recorder = make_drawing_things(normalized_quartic)

if __name__ == "__main__":
	optimized_population, optimized_fitnesses = pop_descent(optimizer, normalized_quartic, new_population, simple_randomizer, observer = recorder)

# dataset = [cat, dog]

# def complex_optimizer(f):
# 	def updater(model):
# 		with tf.gradientTape():
# 			loss = loss_fn(model(dataset))
# 		Adam().optimize(model.trainable_params, loss)
# 		return model
# 	return updater