import random
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import statistics
import tensorflow as tf

def pop_descent(optimizer, normalized_objective, new_individual, randomizer, observer = None, pop_size = 100, iterations = 20):

	# optimizer: np.array(individuals) -> np.array(individual), np.array(floats(0-1))
	# new_population: () -> np.array(individuals)
	# randomizer: np.array(individuals) -> np.array(individuals)
	# observer: np.array(individuals) --> ()
	# pop_size: int (number of individuals)

	replaced_individuals = 30
	normalized_randomness_strength = 0.1

	# creating population of individuals - floats (coordinates)
	population, hist = [], []
	for i in range(pop_size):
		population.append(new_individual())
	population = np.array(population)

	#MAIN LOOP
	for i in tqdm(range(0,iterations), desc = "Running Randomizer"):

		# normalized fitnesses: optimal points close to 1
		fitnesses = normalized_objective(population)

		#graph
		if(observer):
			recorder(population, fitnesses, hist, i, iterations)

		#sorting fitnesses/population from worst to best
		sorted_ind = np.argsort(fitnesses)
		fitnesses = fitnesses[sorted_ind]
		population = population[sorted_ind]

		#calling OPTIMIZER
		population, fitnesses = optimizer(population, normalized_objective)

		#calling SIMPLE RANDOMIZER
		#population[0:replaced_individuals] = randomizer(population[-replaced_individuals:], normalized_randomness_strength)

		#calling WEIGHTED RANDOMIZER
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = fitnesses, k = replaced_individuals)))
		chosen_population = population[chosen_indices]
		randomizer_strength = 1 - (fitnesses[chosen_indices])
		
		population[0:replaced_individuals] = randomizer(chosen_population, randomizer_strength)

	min_recorder(population, hist)
	return population, fitnesses


# Optimizer
def optimizer(pop, objective_function = None):
	#convert np.arrays to tensors
	adam = tf.keras.optimizers.Adam(learning_rate = 1e-2)

	#computing gradient values
	with tf.GradientTape() as tape:
		tensors = tf.Variable(pop)
		fitnesses = 1 - normalized_sin(tensors)

	#convert tensors back to np.arrays, multiply learning rate
	adam.minimize(fitnesses, var_list = [tensors], tape = tape)

	return tensors.numpy(), fitnesses.numpy()


@tf.function
def normalized_quadratic(pop): 
	tensor_values = tf.math.pow(pop, 2)
	return tf.math.subtract(tf.cast(1, tf.float64), tf.math.sigmoid(tensor_values))

# def normalized_quadratic_np(pop):
# 	return normalized_quadratic(pop).numpy()


def normalized_sin_np(x):
	return normalized_sin(x).numpy()

@tf.function
def normalized_sin(pop):
	values = tf.math.sin(pop)/2.0 + 0.5
	return 1 - values


def normalized_quartic(x):
	values = (4*(pop**4))-(3*(pop**2))+(pop*0.01)
	mins, maxs = min(values), max(values)
	return 1 - ((values-mins)/(maxs-mins))


def new_individual():
	coordinate = np.array((random.random()*20. - 10))
	return coordinate

def simple_randomizer(population, normalized_amount):
	#Adds random noise from Gaussian distribution to individual
	mu, sigma = 0, 0.1
	gNoise = (np.random.normal(mu, sigma))*normalized_amount
	return population + gNoise


def make_drawing_things(objective_function):
	def recorder(population, fitnesses, hist, iteration, number_of_iterations):
		hist.append(min(objective_function(population)))
		x = np.array(population)
		y = objective_function(x)

		space = np.linspace(min(x)-10, max(x)+10, len(population)*20)
		ySpace = objective_function(space)

		plt.plot(space, ySpace), plt.scatter(x, y)
		plt.show(block=False), plt.pause(0.0005), plt.close()

		return

	def min_recorder(population, hist):
		#creates graph of minimum function value as individuals are optimized
		optimized_y_vals = list(objective_function(population))
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

recorder, min_recorder = make_drawing_things(normalized_sin_np)

if __name__ == "__main__":
	optimized_population, optimized_fitnesses = pop_descent(optimizer, normalized_sin_np, new_individual, simple_randomizer, observer = recorder)

# dataset = [cat, dog]

# def complex_optimizer(f):
# 	def updater(model):
# 		with tf.gradientTape():
# 			loss = loss_fn(model(dataset))
# 		Adam().optimize(model.trainable_params, loss)
# 		return model
# 	return updater