import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import os
import csv

# cd Documents/
# cd populationDescent/
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m keras_tuner_FMNIST_new

# grad_steps = 25 trials * 2 executions each trial * 938 batches per execution + (20 * 782) for final training = 43000 steps


# def graph_history(history):
# 	integers = [i for i in range(1, (len(history))+1)]

# 	ema = []
# 	avg = history[0]

# 	ema.append(avg)

# 	for loss in history:
# 		avg = (avg * 0.9) + (0.1 * loss)
# 		ema.append(avg)


# 	x = [j * 938 for j in integers]
# 	y = history

# 	# plot line
# 	plt.plot(x, ema[:len(history)])
# 	# plot title/captions
# 	plt.title("Keras Tuner FMNIST")
# 	plt.xlabel("Gradient Steps")
# 	plt.ylabel("Validation Loss")
# 	plt.tight_layout()


# 	print("ema:"), print(ema), print("")
# 	print("x:"), print(x), print("")
# 	print("history:"), print(history), print("")


	
# 	# plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
# 	def save_image(filename):
# 	    p = PdfPages(filename)
# 	    fig = plt.figure(1)
# 	    fig.savefig(p, format='pdf') 
# 	    p.close()

# 	filename = "KerasTuner_FMNIST_progress_with_reg_model4_line.pdf"
# 	save_image(filename)

# 	# plot points too
# 	plt.scatter(x, history, s=20)

# 	def save_image(filename):
# 	    p = PdfPages(filename)
# 	    fig = plt.figure(1)
# 	    fig.savefig(p, format='pdf') 
# 	    p.close()

# 	filename = "KerasTuner_FMNIST_progress_with_reg_model4_with_points.pdf"
# 	save_image(filename)


# 	plt.show(block=True), plt.close()
# 	plt.close('all')


# # history = [] # initialize with for first point

# # unnormalized
# def observer(NN_object, tIndices):
# 	batch_size = 64
# 	tIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

# 	random_batch_FM_validation_images, random_batch_FM_validation_labels = validation_images[tIndices], validation_labels[tIndices]

# 	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 	test_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))
# 	# print(test_loss)
# 	# ntest_loss = 1/(1+test_loss)

# 	return test_loss



# Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

sample_shape = train_images[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Reshape data
train_images = train_images.reshape(len(train_images), input_shape[0], input_shape[1], input_shape[2])
test_images  = test_images.reshape(len(test_images), input_shape[0], input_shape[1], input_shape[2])

# normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

# splitting data into validation/test set
validation_images, validation_labels = test_images[0:5000], test_labels[0:5000]
test_images, test_labels = test_images[5000:], test_labels[5000:]




def build_model(hp):

	model = keras.Sequential()
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape = (28, 28, 1)))
	model.add(tf.keras.layers.Conv2D(128,  kernel_size = 3, strides=(2,2), dilation_rate=(1,1), activation='relu'))
	model.add(tf.keras.layers.Conv2D(256,  kernel_size = 3, dilation_rate=(1,1), activation='relu'))

	
	model.add(tf.keras.layers.Flatten())

	hp_reg = hp.Float("reg_term", min_value=1e-5, max_value=1e-1)

	model.add(tf.keras.layers.Dense(1024, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=hp_reg)))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(10, activation = "softmax"))

	hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

	model.compile(
	    optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
	    loss=keras.losses.SparseCategoricalCrossentropy(),
	    metrics=["accuracy"],
	)

	return model
 


# seed:
def set_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)

def set_global_determinism(seed):
	set_seeds(seed=seed)

	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)


SEED = [5]
# SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]
# s = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

for seed in SEED:  

	set_global_determinism(seed=seed)
	print(seed), print("") 

	import time
	start_time = time.time()  


	max_trials = 25
	model_num = "4 with reg"


	# define tuner
	print("random search")
	tuner = keras_tuner.RandomSearch(
	    hypermodel=build_model,
	    objective="val_accuracy",
	    max_trials=max_trials,
	    executions_per_trial=2,
	    overwrite=True,
	    project_name="FMNIST: %s" % SEED
	)


	# search
	tuner.search(train_images, train_labels, validation_data=(validation_images, validation_labels), batch_size=64)

	# retrieve and train best model
	best_hps = tuner.get_best_hyperparameters(5)
	model = build_model(best_hps[0])


	# Use early stopping
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


	# TRAIN Model
	print("")
	print("TRAINING")
	train_epochs = 20
	hist = model.fit(train_images, train_labels, batch_size=64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])

	# getting history
	print("history"), print(hist.history["val_loss"])
	grad_steps = [i * 936 for i in hist.history['val_loss']]
	print(""), print("grad_steps"), print(grad_steps)
	# graph_history(hist.history["val_loss"])

	time_lapsed = time.time() - start_time


	# evaluating model on test and train data
	batch_size = 64

	np.random.seed(0)
	eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
	random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = train_images[eIndices], train_labels[eIndices], test_images[eIndices], test_labels[eIndices]

	print(""), print(""), print("Evaluating models on test data after randomization")

	# evaluating on train, test images
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy()
	# train_loss = lossfn(random_batch_train_labels, model(random_batch_train_images))
	# test_loss = lossfn(random_batch_test_labels, model(random_batch_test_images))

	train_loss = model.evaluate(random_batch_train_images, random_batch_train_labels)[0]
	test_loss = model.evaluate(random_batch_test_images, random_batch_test_labels)[0]

	print("unnormalized train loss: %s" % train_loss)
	print("unnormalized test loss: %s" % test_loss)
	# print("normalized (1/1+loss) test loss: %s" % ntest_loss)



	model_num = "4_with_reg"


	# writing data to excel file
	data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, seed]]

	with open('/Users/abhi/Documents/research_data/keras_tuner_random_search_FMNIST.csv', 'a', newline = '') as file:
	    writer = csv.writer(file)
	    writer.writerows(data)





