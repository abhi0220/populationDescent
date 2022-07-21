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

def pop_descent(optimizer, new_population, randomizer, pop_size = 20, iterations = 10):

	# optimizer: (individual -> scalar) -> (individual -> individual)
	# normalized_objective: individual -> (float0-1)
	# new_individual: () -> individual
	# randomizer: (individual, float0-1) -> individual
	# observer: () -> graph
	# pop_size: int (number of individuals)

	number_of_replaced_individuals = 6
	#creating population of individual NN models
	population = new_population(pop_size)

#artificial_selection
	for i in tqdm(range(iterations), desc = "Iterations"):

		#calling OPTIMIZER
		population, normalized_training_loss = optimizer(population)

		#sorting losses
		sorted_ind = np.argsort(normalized_training_loss)
		#print(sorted_ind)
		normalized_training_loss = normalized_training_loss[sorted_ind] #worst to best
		population = population[sorted_ind] #worst to best

		#choosing individuals from weighted distribution (using accuracies)
		chosen_indices = np.array((random.choices(np.arange(population.shape[0]), weights = normalized_training_loss, k = number_of_replaced_individuals)))
		#print(chosen_indices)
		chosen_population = population[chosen_indices]
		#print(chosen_population)
		randomizer_strength = 1 - (normalized_training_loss[chosen_indices])
		#print(randomizer_strength)

		#calling WEIGHTED RANDOMIZER
		population[0:number_of_replaced_individuals] = randomizer(chosen_population, randomizer_strength, i)
		# (manual loss) if replacing best with randomized versions, works very well, no repeat losses in evaluator

		#evaluating loss, accuracies after randomization
	evaluator(population)

	return 


# Gradient Descent Optimizer
def NN_optimizer(n):
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

		history = n[g].fit(random_batch_train_images, random_batch_train_labels, batch_size = batch_size, epochs = 1)
		
		lHistory = history.history['loss'] # if epochs = 1
		loss_history.append(lHistory) # if epochs = 1

		aHistory = history.history['accuracy'] # if epochs = 1
		acc_history.append(aHistory) # if epochs = 1

		# print("") # if epochs > 1
		# lHistory = history.history['loss'][-1] # if epochs > 1
		# loss_history.append(lHistory) # if epochs > 1

		# aHistory = history.history['accuracy'][-1] # if epochs > 1
		# acc_history.append(aHistory) # if epochs > 1


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

#need to lower randomizer strnegth after certain amount of iterations
def NN_randomizer(original_population, normalized_amount, iteration):
	print("")
	population_copy = []
	for z in range(len(original_population)):
		model_clone = tf.keras.models.clone_model(original_population[z])
		model_clone.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])

		model_clone.set_weights(np.array(original_population[z].get_weights()))

		mu, sigma = 0, (5e-5) * (1/(iteration+1))
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
	indices = np.random.random_integers(59999, size = (batch_size*50, ))

	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	for g in range(len(n)):
		
		random_batch_train_images = train_images[indices]
		random_batch_train_labels = train_labels[indices]

		model_loss = lossfn(random_batch_train_labels, n[g](random_batch_train_images))
		#print(""), print("lossfn"), print(model_loss)
		normalized_training_loss.append(1/(1+(model_loss)))
		#print(""), print("loss normalized"), print(normalized_training_loss), print("")

	normalized_training_loss = np.array(normalized_training_loss)
	print(""), print(normalized_training_loss)

	return n, normalized_training_loss

def NN_randomizer_manual_loss(original_population, normalized_amount):
	print("")
	population_copy = []
	for z in range(len(original_population)):
		model_clone = tf.keras.models.clone_model(original_population[z])
		# model_clone.compile(optimizer='adam',
  #         		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #         		metrics=['accuracy'])
		model_clone.set_weights(np.array(original_population[z].get_weights()))
		#print(model_clone.get_weights())

		mu, sigma = 0, 1e-3
		gNoise = (np.random.normal(mu, sigma))*(normalized_amount[z])
		#print(gNoise)
		weights = np.array((original_population[z].get_weights()))

		randomized_weights = weights + gNoise

		model_clone.set_weights(randomized_weights)

		#print(""), print("model_clone after randomization"), print(model_clone.get_weights())
		population_copy.append(model_clone)
	population_copy = np.array(population_copy)

	return population_copy

def NN_compiler(n):
	for z in range(len(n)):
		n[z].compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	return

def evaluator(n):
	NN_compiler(n) # only if using manual loss optimizer/randomizer
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