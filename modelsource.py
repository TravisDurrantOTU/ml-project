import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn import metrics

def confusion_matrix(predictions, true_labels):
    cm = metrics.confusion_matrix(true_labels, predictions)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [0,1,2,3,4,5,6,7,8,9])
    display.plot()
    plt.show() 
    return

def train_model(filters = 6, 
                conv_neigh=(5, 5), 
                max_neigh=(2, 2), 
                opt = 'adam', 
                activation = 'relu', 
                metrics=['accuracy'],
                epochs = 5,
                batch_size = 64,
                validation_split = 0.1,
                learning_rate = 0.001
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
    model.optimizer.learning_rate.assign(learning_rate)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # Predict on the test dataset
    predictions = model.predict(x_test)

    return predictions, test_loss, test_acc, confusion_matrix(np.argmax(a=predictions, axis=1), np.array(y_test))

def naive_model(filters = 14, 
                conv_neigh=(7, 7), 
                max_neigh=(2, 2), 
                opt = 'rmsprop', 
                activation = 'tanh', 
                metrics=['accuracy'],
                epochs = 15,
                batch_size = 64,
                validation_split = 0.1,
                learning_rate = 0.001
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
    model.optimizer.learning_rate.assign(learning_rate)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # Predict on the test dataset
    predictions = model.predict(x_test)

    return predictions, test_loss, test_acc, confusion_matrix(np.argmax(a=predictions, axis=1), np.array(y_test))

def greedy_model(
                filters = 16, 
                conv_neigh=(7, 7), 
                max_neigh=(3, 3), 
                opt = 'rmsprop', 
                activation = 'relu', 
                metrics=['accuracy'],
                epochs = 15,
                batch_size = 64,
                validation_split = 0.1,
                learning_rate = 0.001
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
    model.optimizer.learning_rate.assign(learning_rate)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # Predict on the test dataset
    predictions = model.predict(x_test)

    # Find misclassified images
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    for i in range(len(x_test)):
        true_label = y_test[i]
        predicted_label = np.argmax(predictions[i])  # Predicted label
        if predicted_label != true_label:  # If prediction is incorrect
            misclassified_images.append(x_test[i])
            misclassified_labels.append(true_label)
            misclassified_preds.append(predicted_label)

    # Print out misclassified images
    for i in range(min(5, len(misclassified_images))):  # Limiting to 5 images
        plt.imshow(misclassified_images[i].reshape(28, 28), cmap='gray')
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title(f"True Label: {misclassified_labels[i]}, Predicted: {misclassified_preds[i]}")
        plt.show()

    return predictions, test_loss, test_acc, confusion_matrix(np.argmax(a=predictions, axis=1), np.array(y_test))

greedy_model()