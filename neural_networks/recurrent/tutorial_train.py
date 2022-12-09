#!/usr/bin/env python3

import matplotlib.pyplot as plt
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
    data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
    target = [(i+5)/100 for i in range(100)]

    data = np.array(data, dtype=float)
    target = np.array(target, dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)

    model = Sequential()
    model.add(LSTM(1, batch_input_shape=(None, None, 1), return_sequences=False))
    # Dropout(0.2)
    # model.add(LSTM(1))
    # model.add(Dense(10, activation="softmax"))

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    model.summary()

    # print(x_train.shape)
    # print(y_train.shape)
    fit_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=400)
    # fit_history = model.fit(x=x_train, y=y_train, validation_split=0.7, epochs=25, verbose=2, batch_size=32)

    # fig = plt.figure()
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(fit_history.history['accuracy'])
    # plt.plot(fit_history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(fit_history.history['loss'])
    # plt.plot(fit_history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    #
    # plt.show()

    results = model.predict(x_test)

    # results = np.argmax(predicted_values, axis=1, out=None)
    # y_results = np.argmax(y_test, axis=1, out=None)

    plt.scatter(range(results.shape[0]), results, color='r')
    plt.scatter(range(results.shape[0]), y_test, color='g')
    plt.show()

    data = [[[(i + j) / 100] for i in range(6)] for j in range(100)]
    target = [(i + 6) / 100 for i in range(100)]

    data = np.array(data, dtype=float)
    target = np.array(target, dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
    fit_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=400)
    # model.save("myModel")

    results = model.predict(x_test)

    # results = np.argmax(predicted_values, axis=1, out=None)
    # y_results = np.argmax(y_test, axis=1, out=None)

    plt.scatter(range(results.shape[0]), results, color='r')
    plt.scatter(range(results.shape[0]), y_test, color='g')
    plt.show()





