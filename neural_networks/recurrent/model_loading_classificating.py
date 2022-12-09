#!/usr/bin/env python3

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout  # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage

if __name__ == '__main__':
    time_steps = 20
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    # times =
    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_test = test_data.shape[0]

    x_test_original = np.reshape(test_data[:, :-1], (n_test, 50, 13))
    x_test_original = x_test_original[:, :, 1:]
    y_test_original = to_categorical(test_data[:, -1])

    x_test = x_test_original[:, 0:time_steps, :]

    model = keras.models.load_model("myModel")

    predicted_values = model.predict(x_test)

    results = np.argmax(predicted_values, axis=1, out=None)
    y_results = np.argmax(y_test_original, axis=1, out=None)

    # plt.scatter(range(results.shape[0]), results, color='r')
    # plt.scatter(range(results.shape[0]), y_results, color='g')
    # plt.show()

    cm = confusion_matrix(y_true=y_results, y_pred=results)
    cm_true = cm / cm.astype(float).sum(axis=1)
    blues = plt.cm.Blues
    cm_mean = cm_true * 100
    title = "Confusion Matrix (%) - Mean (" + str(
        time_steps) + " time steps) \n LSTM "
    plot_confusion_matrix_percentage(confusion_matrix=cm_mean, display_labels=labels, cmap=blues,
                                     title=title, decimals=.2)

    plt.show()
    # print("cm")
    # print(cm)
