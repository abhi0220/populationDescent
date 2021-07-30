import random
import warnings
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import statistics
import tensorflow as tf

warnings.filterwarnings("ignore", category=DeprecationWarning)

def pop_descent(optimizer, new_individual, randomizer, pop_size = 5, iterations = 5):

	# optimizer: (individual -> scalar) -> (individual -> individual)
	# normalized_objective: individual -> (float0-1)
	# new_individual: () -> individual
	# randomizer: (individual, float0-1) -> individual
	# observer: () -> graph
	# pop_size: int (number of individuals)

	number_of_replaced_individuals = 1
	# creating population of individuals - floats (coordinates)
	population, hist = [], []
	for i in range(pop_size):
		population.append(new_individual())
	population = np.array(population)

	for i in (range(iterations)):
		print(""), print("optimizer: model.fitting"), print("")

		#calling OPTIMIZER
		#optimizer(population)

		#evaluating loss, accuracies/fitnesses
		test_loss, test_acc = [], []
		print(""), print("after optimization evaluation:")

		indices = np.random.random_integers(9999, size = (1000,))
		for i in (range(pop_size)):
			random_batch_test_images = test_images[indices]
			random_batch_test_labels = test_labels[indices]
			ntest_loss, ntest_acc = population[i].evaluate(random_batch_test_images, random_batch_test_labels, verbose=1)
			test_loss.append(ntest_loss)
			test_acc.append(ntest_acc)

		avg_loss = np.mean(test_loss)
		avg_acc = np.mean(test_acc)

		print(""), print("avg_loss: %s" % avg_loss), print("avg_acc: %s" % avg_acc), print("")

		#normalizing accuracies, population for random selection
		test_acc = np.array(test_acc)

		#sorting accuracies, 
		sorted_ind = np.argsort(test_acc)
		test_acc = test_acc[sorted_ind]
		population = population[sorted_ind]

		#choosing individuals from weighted distribution (using accuracies)
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = test_acc, k = number_of_replaced_individuals)))
		chosen_population = population[chosen_indices]
		randomizer_strength = (1 - (test_acc[chosen_indices]))

		#calling WEIGHTED RANDOMIZER
		population[0:number_of_replaced_individuals] = randomizer(chosen_population, randomizer_strength)

		#evaluating loss, accuracies after randomization
		test_loss, test_acc = [], []
		print(""), print("after randomization fitting:")
		nindices = np.random.random_integers(9999, size = (1000,))
		for j in (range(len(population))):
			random_batch_test_images = test_images[indices]
			random_batch_test_labels = test_labels[indices]
			ntest_loss, ntest_acc = population[j].evaluate(random_batch_test_images, random_batch_test_labels, verbose=1)
			test_loss.append(ntest_loss)
			test_acc.append(ntest_acc)

		avg_loss = np.mean(test_loss)
		avg_acc = np.mean(test_acc)

		print(""), print("avg_loss: %s" % avg_loss), print("avg_acc: %s" % avg_acc), print("")

	return 


# Gradient Descent Optimizer
def NN_optimizer(n):
	for i in (range(len(n))):
		indices = np.random.random_integers(59999, size = (2500,))
		random_batch_train_images = train_images[indices]
		random_batch_train_labels = train_labels[indices]

		n[i].fit(random_batch_train_images, random_batch_train_labels, epochs=1)
	return n

def new_individual():
	model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
	])
	model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	return model

def NN_randomizer(population, normalized_amount):
	for i in range(len(population)):
		mu, sigma = 0, 0.1
		gNoise = (np.random.normal(mu, sigma))*normalized_amount[i]*0.001
		weights = (np.array((population[i].get_weights())))
		randomized_weights = np.add(weights, gNoise)
		population[i].set_weights((randomized_weights))

	return population


#importing data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

if __name__ == "__main__":
	pop_descent(NN_optimizer, new_individual, NN_randomizer)

# dataset = [cat, dog]

# def complex_optimizer(f):
# 	def updater(model):
# 		with tf.gradientTape():
# 			loss = loss_fn(model(dataset))
# 		Adam().optimize(model.trainable_params, loss)
# 		return model
# 	return updater