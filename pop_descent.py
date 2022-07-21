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

def pop_descent(optimizer, randomizer, new_population, pop_size = 5, number_of_replaced_individuals = 2, iterations = 10, final_observer = None, recorder = None, normalized_objective = None, normalized_randomness_strength = None):

	# optimizer: (individual -> scalar) -> (individual -> individual)
	# normalized_objective: individual -> (float0-1)
	# new_individual: () -> individual
	# randomizer: (individual, float0-1) -> individual
	# observer: () -> graph
	# pop_size: int (number of individuals)

	#creating population of individual NN models
	population, hist = new_population(pop_size)

#artificial_selection
	for i in tqdm(range(iterations), desc = "Iterations"):

		#calling OPTIMIZER
		population, Lfitnesses, fitnesses = optimizer(population)

		#sorting losses
		sorted_ind = np.argsort(fitnesses)
		fitnesses = fitnesses[sorted_ind] #worst to best
		population = population[sorted_ind] #worst to best

		#choosing individuals from weighted distribution (using accuracies)
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = fitnesses, k = number_of_replaced_individuals)))
		chosen_population = population[chosen_indices]
		randomizer_strength = 1 - (fitnesses[chosen_indices])

		#calling WEIGHTED RANDOMIZER
		population[0:number_of_replaced_individuals] = randomizer(chosen_population, randomizer_strength, i)

		#evaluating loss(NN)/graph(coordinates)
	if (recorder):
		recorder(population, hist)

	#graph of loss optimization
	if (final_observer):
		final_observer(population, hist)


	return population, Lfitnesses, fitnesses

