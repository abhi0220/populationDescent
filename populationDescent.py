import random
import warnings
import matplotlib.pyplot as plt
import scipy
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import statistics
import tensorflow as tf
# from keras.callbacks import History

# warnings.filterwarnings("ignore", category=DeprecationWarning)

def populationDescent(Parameters, number_of_replaced_individuals, iterations):

	# optimizer: (individual -> scalar) -> (individual -> individual)
	# normalized_objective: individual -> (float0-1)
	# new_individual: () -> individual
	# randomizer: (individual, float0-1) -> individual
	# observer: () -> graph
	# pop_size: int (number of individuals)

	#creating population of individual NN models
	hist = []
	# population = new_population(pop_size)

#artificial_selection
	for i in tqdm(range(iterations), desc = "Iterations"):

		#calling OPTIMIZER
		lFitnesses, vFitnesses = Parameters.optimizer(Parameters.population)

		if Parameters.CV_selection==False:
			#sorting losses (training)
			sorted_ind = np.argsort(lFitnesses)
			lFitnesses = lFitnesses[sorted_ind] #worst to best
			Parameters.population = Parameters.population[sorted_ind] #worst to best

			# #choosing individuals from weighted distribution (training)
			chosen_indices = np.array((random.choices(np.arange(Parameters.population.shape[0]), weights = lFitnesses, k = number_of_replaced_individuals)))
			chosen_population = Parameters.population[chosen_indices]
			randomizer_strength = 1 - (lFitnesses[chosen_indices])

		if Parameters.CV_selection==True:
			#sorting losses (validation)
			sorted_ind = np.argsort(vFitnesses)
			vFitnesses = vFitnesses[sorted_ind] #worst to best
			Parameters.population = Parameters.population[sorted_ind] #worst to best

			#choosing individuals from weighted distribution (validation)
			chosen_indices = np.array((random.choices(np.arange(Parameters.population.shape[0]), weights = vFitnesses, k = number_of_replaced_individuals)))
			chosen_population = Parameters.population[chosen_indices]
			randomizer_strength = 1 - (vFitnesses[chosen_indices])

		# observing optimization progress
		if i%(Parameters.rr)==0:
				if i!=(iterations-1):
					print(""), print("observing"), print("..."), print("")
					Parameters.observer(Parameters.population, Parameters.history)

		#calling WEIGHTED RANDOMIZER
		if (Parameters.randomization)==True:
			if i%(Parameters.rr)==0:
				if i!=(iterations-1):
					print("")
					Parameters.population[0:number_of_replaced_individuals] = Parameters.randomizer(chosen_population, randomizer_strength)

	return Parameters.population, lFitnesses, vFitnesses, Parameters.history

