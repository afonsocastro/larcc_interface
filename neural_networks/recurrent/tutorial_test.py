#!/usr/bin/env python3

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, MaxPooling2D, Dropout # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage

if __name__ == '__main__':
    model = keras.models.load_model("myModel")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    cut_value = 60000

    x_train = x_train[0:cut_value]
    x_test = x_test[0:cut_value]
    y_train = y_train[0:cut_value]
    y_test = y_test[0:cut_value]

    predicted_values = model.predict(x_test)

    results = np.argmax(predicted_values, axis=1, out=None)
    y_results = np.argmax(y_test, axis=1, out=None)

    # plt.scatter(range(results.shape[0]), results, color='r')
    # plt.scatter(range(results.shape[0]), y_results, color='g')
    # plt.show()

    cm = confusion_matrix(y_true=y_results, y_pred=results)
    print("cm")
    print(cm)
