import random
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import statistics
import tensorflow as tf

from pop_descent import pop_descent

# OPTIMIZER
def optimizer(pop, normalized_objective = None):
	#convert np.arrays to tensors
	adam = tf.keras.optimizers.Adam(learning_rate = 1e-3)
	#computing gradient values
	with tf.GradientTape() as tape:
		tensors = tf.Variable(pop)
		fitnesses = 1 - normalized_objective(tensors)

	#convert tensors back to np.arrays, multiply learning rate
	adam.minimize(fitnesses, var_list = [tensors], tape = tape)

	return tensors.numpy(), 1 - fitnesses.numpy()

# RANDOMIZER
def simple_randomizer(population, normalized_amount, iteration = None):
	#Adds random noise from Gaussian distribution to individual
	mu, sigma = 0, 0.1
	gNoise = (np.random.normal(mu, sigma))*normalized_amount
	return population + gNoise

# NEW POPULATION
def new_population(pop_size):
	population = []
	hist = []
	for i in range(pop_size):
		coordinate = np.array((random.random()*20. - 10))
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
def normalized_quartic(pop):
	tensor_values = (4*tf.math.pow(pop, 4))-(3*(tf.math.pow(pop, 2))+(0.01*pop))
	return tf.math.divide(tf.cast(1, tf.float64), tf.math.add(tf.cast(1, tf.float64), tensor_values))

def make_drawing_things(objective_function):
	def graph_recorder(population, hist):
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

	return graph_recorder, min_recorder

graph_recorder, min_recorder = make_drawing_things(normalized_quartic)


# FUNCTIONS FOR NN IMPLEMENTATION

# Gradient Descent Optimizer
def NN_optimizer(n, normalized_objective = None):
	print("Optimizing (model.fit())")
	loss_history = []
	acc_history = []
	batch_size = 64
	#indices = np.arange(train_images.shape[0])
	indices = np.random.random_integers(59999, size = (batch_size*5, ))
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	normalized_training_loss = []
	for g in (range(len(n))):
		random_batch_train_images = train_images[indices]
		random_batch_train_labels = train_labels[indices]
		# Actual optimization on training data
		history = n[g].fit(random_batch_train_images, random_batch_train_labels, batch_size = batch_size, epochs = 1)

		lHistory = history.history['loss'] # if epochs = 1
		aHistory = history.history['accuracy'] # if epochs = 1

		# lHistory = history.history['loss'][-1] # if epochs > 1
		# aHistory = history.history['accuracy'][-1] # if epochs > 1

		loss_history.append(lHistory)
		acc_history.append(aHistory)

	loss_history = np.array(loss_history)
	avg_loss = np.mean(loss_history)
	print("avg_loss: %s" % avg_loss)

	acc_history = np.array(acc_history)
	avg_acc = np.mean(acc_history)
	print("avg_acc: %s" % avg_acc), print("")

	#normalizing losses, population for random selection
	normalized_training_loss = np.array((1/(1+loss_history))) #higher is better
	normalized_training_loss = normalized_training_loss.reshape([len(normalized_training_loss), ])

	return n, normalized_training_loss

def NN_randomizer(original_population, normalized_amount, iteration = None):
	print("")
	population_copy = []
	for z in range(len(original_population)):
		model_clone = tf.keras.models.clone_model(original_population[z])
		model_clone.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
		model_clone.set_weights(np.array(original_population[z].get_weights()))

		if (iteration):
			decreasing_factor = (1/(iteration+1))
		else:
			decreasing_factor = 1

		mu, sigma = 0, (5e-5) * decreasing_factor
		gNoise = (np.random.normal(mu, sigma))*(normalized_amount[z])

		weights = np.array((model_clone.get_weights()))
		randomized_weights = weights + gNoise
		model_clone.set_weights((randomized_weights))
		population_copy.append(model_clone)

	return np.array(population_copy)

def NN_optimizer_manual_loss(n):
	batch_size = 64
	normalized_training_loss = []
	#indices = np.arange(train_images.shape[0])

	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	for g in range(len(n)):
		indices = np.random.random_integers(59999, size = (batch_size, ))
		random_batch_train_images = train_images[indices]
		random_batch_train_labels = train_labels[indices]

		model_loss = lossfn(random_batch_train_labels, n[g](random_batch_train_images))
		normalized_training_loss.append(1/(1+(model_loss)))

	normalized_training_loss = np.array(normalized_training_loss)
	print(normalized_training_loss)

	return n, normalized_training_loss

def NN_randomizer_manual_loss(original_population, normalized_amount):
	print("")
	population_copy = []
	for z in range(len(original_population)):
		model_clone = tf.keras.models.clone_model(original_population[z])

		model_clone.set_weights(np.array(original_population[z].get_weights()))
		# print(model_clone.get_weights())

		mu, sigma = 0, 0.1
		gNoise = (np.random.normal(mu, sigma))*(normalized_amount[z])

		weights = np.array((original_population[z].get_weights()))
		randomized_weights = weights + gNoise
		model_clone.set_weights(randomized_weights)
		# print(""), print("model_clone after randomization")
		# print(model_clone.get_weights())
		population_copy.append(model_clone)

	return np.array(population_copy)


def NN_compiler(n):
	for z in range(len(n)):
		n[z].compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	return

def evaluator(n, hist):
	#NN_compiler(n) # only if using manual loss optimizer/randomizer
	test_loss, test_acc = [], []
	batch_size = 64
	indices = np.random.random_integers(9999, size = (batch_size*25, ))
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
	hist.append(avg_loss)
	print(""), print("avg_loss: %s" % avg_loss), print("avg_acc: %s" % avg_acc), print(""), print("")

	return

def new_NN_population(pop_size):
	population, hist = [], []
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
	return population, hist


#importing data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0




if __name__ == "__main__":
	optimized_population, optimized_fitnesses = pop_descent(optimizer = optimizer, randomizer = simple_randomizer, new_population = new_population, pop_size = 1000, number_of_replaced_individuals = 200, iterations = 20, final_observer = min_recorder, recorder = graph_recorder, normalized_objective = normalized_quartic, normalized_randomness_strength = None)
	#optimized_population, optimized_fitnesses = pop_descent(NN_optimizer, NN_randomizer, new_NN_population, pop_size = 10, number_of_replaced_individuals = 3, iterations = 10, final_observer = None, recorder = evaluator, normalized_objective = None, normalized_randomness_strength = None)
