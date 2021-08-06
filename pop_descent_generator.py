import random
import warnings
import matplotlib.pyplot as plt
import scipy
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import statistics
import tensorflow as tf
from keras.callbacks import History

warnings.filterwarnings("ignore", category=DeprecationWarning)

#from pop_descent_functions import optimizer, simple_randomizer, new_population, normalized_sin, min_recorder, graph_recorder, normalized_sin, normalized_quartic
from pop_descent_functions import NN_optimizer, NN_randomizer, new_NN_population

def pop_descent(optimizer, randomizer, new_population, pop_size = 5, number_of_replaced_individuals = 2, iterations = 10, final_observer = None, recorder = None, normalized_objective = None, normalized_randomness_strength = None):

	#artificial_selection
	def artificial_selection(population):

		#calling OPTIMIZER
		population, fitnesses = optimizer(population, normalized_objective)

		#sorting losses
		sorted_ind = np.argsort(fitnesses)
		fitnesses = fitnesses[sorted_ind] #worst to best
		population = population[sorted_ind] #worst to best

		#choosing individuals from weighted distribution (using accuracies)
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = fitnesses, k = number_of_replaced_individuals)))
		chosen_population = population[chosen_indices]
		randomizer_strength = 1 - (fitnesses[chosen_indices])

		#calling WEIGHTED RANDOMIZER
		population[0:number_of_replaced_individuals] = randomizer(chosen_population, randomizer_strength)

		#evaluating loss(NN)/graph(coordinates)
		if (recorder):
			recorder(population, hist)

		#graph of loss optimization
		if (final_observer):
			final_observer(population, hist)

		return population

	return artificial_selection


if __name__ == "__main__":

	pop_size = 10
	#artificial_selection = pop_descent(optimizer = optimizer, randomizer = simple_randomizer, new_population = new_population, pop_size = pop_size, number_of_replaced_individuals = 30, final_observer = None, recorder = graph_recorder, normalized_objective = normalized_quartic)
	artificial_selection = pop_descent(optimizer = NN_optimizer, randomizer = NN_randomizer, new_population = new_NN_population, pop_size = pop_size, number_of_replaced_individuals = 2, final_observer = None, recorder = None, normalized_objective = None)


	# population, hist = new_population(pop_size)
	population, hist = new_NN_population(pop_size)

	for i in range(20):
		population = artificial_selection(population)

