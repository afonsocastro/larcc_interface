#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout  # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage


def create_model():
    model = Sequential()
    model.add(LSTM(64, batch_input_shape=(None, None, 12), return_sequences=True))
    # model.add(LSTM(64, input_shape=(50, 13)))
    Dropout(0.2)
    model.add(LSTM(64))
    model.add(Dense(4, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    model.summary()

    return model


if __name__ == '__main__':

    # times =
    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    validation_split = 0.3
    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1 - validation_split)
    n_test = test_data.shape[0]

    x_train_original = np.reshape(training_data[:, :-1], (training_data.shape[0], 50, 13))
    x_train_original = x_train_original[:, :, 1:]
    y_train_original = to_categorical(training_data[:, -1])
    x_test_original = np.reshape(test_data[:, :-1], (n_test, 50, 13))
    x_test_original = x_test_original[:, :, 1:]
    y_test_original = to_categorical(test_data[:, -1])

    model = create_model()

    final_x_train = []
    final_x_test = []

    final_y_test = []
    final_y_train = []
    # total_fit_history = []
    for n in range(5, 50):
        x_train = x_train_original[:, 0:n, :]
        x_test = x_test_original[:, 0:n, :]
        callback = EarlyStopping(monitor='val_loss', patience=10)
        fit_history = model.fit(x=x_train, y=y_train_original, validation_split=validation_split, epochs=100,
                                shuffle=True)
        print("fit_history.history['accuracy']")
        print(type(fit_history.history['accuracy']))
        final_x_test.append(x_test)
        final_y_test.append(y_test_original)
        # total_fit_history.append(fit_history)

    model.save("myModel")

    # for test in range(0, len(final_y_test)):
    #     predicted_values = model.predict(final_x_test[test])
    #     results = np.argmax(predicted_values, axis=1, out=None)
    #     y_results = np.argmax(final_y_test[test], axis=1, out=None)
    #
    #     cm = confusion_matrix(y_true=y_results, y_pred=results)
    #     print("cm")
    #     print(cm)
