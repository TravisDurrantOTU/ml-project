import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def confusion_matrix(predictions, x_test):
    return 0

def train_model(filters = 6, 
                conv_neigh=(5, 5), 
                max_neigh=(2, 2), 
                opt = 'adam', 
                activation = 'relu', 
                metrics=['accuracy'],
                epochs = 5,
                batch_size = 64,
                validation_split = 0.1
                ):
    # load the data from an easier source than actually reading it
    (x_train, y_train),(x_test,y_test) = mnist.load_data()

    # preprocessing
    # the data is actually already preprocessed for this dataset

    # reshaping the data
    # not doing any trimming or scaling 
    # just changing a (28, 28) to be explicitly (28, 28, 1) rather than implicit
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


    # Define the model
    model = models.Sequential([
        layers.Conv2D(filters, conv_neigh, activation=activation, input_shape=(28, 28, 1)),
        layers.MaxPooling2D(max_neigh),
        layers.Flatten(),
        layers.Dense(64, activation=activation),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=metrics)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # Predict on the test dataset
    predictions = model.predict(x_test)

    return predictions, test_loss, test_acc, confusion_matrix(predictions, x_test)

print('this shouldn\'t be running like this')