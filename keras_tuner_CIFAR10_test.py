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
# python3 -m keras_tuner_CIFAR10_test


SEED = 60

# seed:
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)



s = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

set_global_determinism(seed=SEED)
print(SEED), print("")


# CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

validation_images, validation_labels = test_images[0:5000], test_labels[0:5000]
test_images, test_labels = test_images[5000:], test_labels[5000:]



class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):

        # model_num = "6 CIFAR with_reg"
        # model = tf.keras.Sequential([
        # tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)),
        # # tf.keras.layers.BatchNormalization(),
        
        # # tf.keras.layers.Dropout(0.2),
        
        # tf.keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'),
        # # tf.keras.layers.BatchNormalization(),
        
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu'),
        # # tf.keras.layers.BatchNormalization(),
        
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
        # # tf.keras.layers.BatchNormalization(),
        
        # tf.keras.layers.MaxPooling2D((4, 4)),
        # # tf.keras.layers.Dropout(0.2),

        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),

        # tf.keras.layers.Dense(10, activation = "softmax")
        # ])

        """Builds a convolutional model."""
        inputs = keras.Input((32, 32, 3))
        x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu')(inputs)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu')(x)
        # x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(3,3), dilation_rate=(1,1), activation='relu')(x)
        # x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu')(x)
        # x = keras.layers.MaxPooling2D((4, 4))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=hp.Float("regularization_rate", 1e-5, 1e-1)))(x)
        x = keras.layers.Activation("relu")(x)

        outputs = keras.layers.Dense(10, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)


        hp_learning_rate = hp.Float("learning_rate", 1e-5, 1e-1, sampling="log")

        # custom_optimizer = keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-5, 1e-1, sampling="log"))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


        return model

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # batch_size = hp.Int("batch_size", 4, 260, step=32)
        # batches = hp.Int("batches", 4, 260, step=32)

        batch_size = 64
        batches = 128

        indices = np.random.choice(49999, size = (batch_size*batches, ), replace=False)
        vIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

        val_input_data = (validation_data[0][vIndices], validation_data[1][vIndices])

        train_ds = tf.data.Dataset.from_tensor_slices((train_images[indices], train_labels[indices])).batch(
            batch_size
        )
        validation_data = tf.data.Dataset.from_tensor_slices(val_input_data).batch(
            batch_size
        )


        # Define the optimizer.
        optimizer = keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-5, 1e-1, sampling="log"))
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # The metric to track validation loss.
        epoch_loss_metric = keras.metrics.Mean()

        # Function to run the train step.
        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Function to run the validation step.
        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            # Update the metric.
            epoch_loss_metric.update_state(loss)

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model

        # Record the best validation loss value
        best_epoch_loss = float("inf")

        # epochs = hp.Int("epochs", 1, 12, step=1)
        epochs = 1

        # The custom training loop.
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            # Iterate the training data to run the training step.
            for images, labels in train_ds:
                run_train_step(images, labels)

            # Iterate the validation data to run the validation step.
            for images, labels in validation_data:
                run_val_step(images, labels)

            # Calling the callbacks after epoch.
            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"val_accuracy": epoch_loss})
            epoch_loss_metric.reset_states()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        # Return the evaluation metric value.
        return best_epoch_loss



max_trials = 0

tuner = keras_tuner.RandomSearch(
    objective=keras_tuner.Objective("val_accuracy", "min"),
    max_trials=max_trials,
    hypermodel=MyHyperModel(),
)


import time
start_time = time.time()

# running keras tuner search
tuner.search(x=train_images, y=train_labels, validation_data=(validation_images, validation_labels))

# retreiving best hyperparameters from search
best_hps = tuner.get_best_hyperparameters()[0]


# Build the model with the best hp.
h_model = MyHyperModel()
model = h_model.build(best_hps)
model.summary()

# get best hyperparameters
print(best_hps.values)
# custom_batch_size, custom_batches, custom_learning_rate, custom_epochs = best_hps["batch_size"], best_hps["batches"], best_hps["learning_rate"], best_hps["epochs"]
custom_learning_rate, custom_reg_rate = best_hps["learning_rate"], best_hps["regularization_rate"]

# compiling with custom optimizer
# opt = tf.keras.optimizers.Adam(learning_rate = custom_learning_rate)
# model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# print(model.optimizer.get_config())

# setting a callback to prevent overfitting
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

# retraining model with best hyperparameters
for i in range(5):
    # indices = np.random.choice(49999, size = (custom_batch_size*custom_batches, ), replace=False)
    vIndices = np.random.choice(4999, size = (64*10, ), replace=False)
    val_input_data = (validation_images[vIndices], validation_labels[vIndices])

    # model.fit(train_images[indices], train_labels[indices], batch_size=custom_batch_size, epochs=1, verbose=1)
    model.fit(train_images, train_labels, batch_size=64, epochs=1, verbose=1)
    # model.fit(train_images[indices], train_labels[indices], validation_data=val_input_data, epochs=custom_epochs, callbacks=[callback], verbose=1)


time_lapsed = time.time() - start_time

# evaluating model on test and train data
batch_size = 64

np.random.seed(0)
eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = train_images[eIndices], train_labels[eIndices], test_images[eIndices], test_labels[eIndices]

print(""), print(""), print("Evaluating models on test data after randomization")

# evaluating on train, test images
lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = lossfn(random_batch_train_labels, model(random_batch_train_images))
test_loss = lossfn(random_batch_test_labels, model(random_batch_test_images))

print("unnormalized train loss: %s" % train_loss)
print("unnormalized test loss: %s" % test_loss)
# print("normalized (1/1+loss) test loss: %s" % ntest_loss)



model_num = "6_no_reg"


# writing data to excel file
data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, custom_epochs, custom_batches, custom_batch_size, custom_learning_rate, SEED]]

with open('/Users/abhi/Documents/research_data/keras_tuner_hyperband_FMNIST1.csv', 'a', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows(data)



