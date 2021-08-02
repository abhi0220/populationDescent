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
from keras.callbacks import History

warnings.filterwarnings("ignore", category=DeprecationWarning)

def pop_descent(optimizer, new_population, randomizer, pop_size = 50, iterations = 25):

	# optimizer: (individual -> scalar) -> (individual -> individual)
	# normalized_objective: individual -> (float0-1)
	# new_individual: () -> individual
	# randomizer: (individual, float0-1) -> individual
	# observer: () -> graph
	# pop_size: int (number of individuals)

	number_of_replaced_individuals = 15
	#creating population of individual NN models
	population = new_population(pop_size)


	for i in tqdm(range(iterations), desc = "Iterations"):

		#calling OPTIMIZER
		pop, normalized_training_loss = optimizer(population)

		# #normalizing losses, population for random selection
		# normalized_training_loss = np.array((1/(1+loss_history))) #higher is better
		# normalized_training_loss = normalized_training_loss.reshape([len(normalized_training_loss), ])

		#sorting losses
		sorted_ind = np.argsort(normalized_training_loss)
		normalized_training_loss = normalized_training_loss[sorted_ind] #worst to best

		population = population[(sorted_ind)] #worst to best

		#choosing individuals from weighted distribution (using accuracies)
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = normalized_training_loss, k = number_of_replaced_individuals)))
		chosen_population = population[chosen_indices]
		randomizer_strength = (normalized_training_loss[chosen_indices])

		#calling WEIGHTED RANDOMIZER
		population[0:number_of_replaced_individuals] = randomizer(chosen_population, randomizer_strength)

		#evaluating loss, accuracies after randomization
		evaluator(population)

	return 


# Gradient Descent Optimizer
def NN_optimizer(n):
	print("Optimizing (model.fit())")
	loss_history = []
	batch_size = 64
	#indices = np.arange(train_images.shape[0])
	indices = np.random.random_integers(59999, size = (batch_size*100, ))
	for g in (range(len(n))):
		random_batch_train_images = train_images[indices]
		random_batch_train_labels = train_labels[indices]

		history = n[g].fit(random_batch_train_images, random_batch_train_labels, batch_size = batch_size, epochs=1)

		hist = history.history['loss']
		loss_history.append(hist)

	loss_history = np.array(loss_history)
	avg_loss = np.mean(loss_history)
	print("avg_loss: %s" % avg_loss)

	#normalizing losses, population for random selection
	normalized_training_loss = np.array((1/(1+loss_history))) #higher is better
	normalized_training_loss = normalized_training_loss.reshape([len(normalized_training_loss), ])

	return n, normalized_training_loss

def NN_randomizer(original_population, normalized_amount):
	print("")
	population_copy = []
	for z in range(len(original_population)):
		model_clone = tf.keras.models.clone_model(original_population[z])
		model_clone.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])

		#model_clone.set_weights(original_population[z].get_weights())

		mu, sigma = 0, 0.1
		gNoise = (np.random.normal(mu, sigma))*(normalized_amount[z])*(5e-3)

		weights = np.array((original_population[z].get_weights()))
		randomized_weights = weights + gNoise
		model_clone.set_weights((randomized_weights))
		population_copy.append(model_clone)

	return np.array(population_copy)

def evaluator(n):
	test_loss, test_acc = [], []
	batch_size = 64
	indices = np.random.random_integers(9999, size = (batch_size*50, ))
	#indices = np.arange(test_images.shape[0])
	print("Evaluating models on test data after randomization")
	for j in (range(len(n))):
		random_batch_test_images = test_images[indices]
		random_batch_test_labels = test_labels[indices]
		ntest_loss, ntest_acc = n[j].evaluate(random_batch_test_images, random_batch_test_labels, batch_size = batch_size, verbose=1)
		test_loss.append(ntest_loss)
		test_acc.append(ntest_acc)

	avg_loss = np.mean(test_loss)
	avg_acc = np.mean(test_acc)
	print(""), print("avg_loss: %s" % avg_loss), print("avg_acc: %s" % avg_acc), print(""), print("")

	return

def new_population(pop_size):
	population = []
	for i in range(pop_size):
		model = tf.keras.Sequential([
	    tf.keras.layers.Flatten(input_shape=(28, 28)),
	    tf.keras.layers.Dense(128, activation='relu'),
	    tf.keras.layers.Dense(10)
		])

		model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])

		population.append(model)

	population = np.array(population)
	return population



#importing data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

if __name__ == "__main__":
	pop_descent(NN_optimizer, new_population, NN_randomizer)

# dataset = [cat, dog]

# def complex_optimizer(f):
# 	def updater(model):
# 		with tf.gradientTape():
# 			loss = loss_fn(model(dataset))
# 		Adam().optimize(model.trainable_params, loss)
# 		return model
# 	return updater