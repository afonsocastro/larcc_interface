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

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(type(x_train))
    print(type(x_test))
    print(type(y_train))
    print(type(y_test))

    # exit(0)
    model = Sequential()
    model.add(LSTM(64, input_shape=(28, 28), return_sequences=True))
    Dropout(0.2)
    model.add(LSTM(64))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    model.summary()

    print(x_train.shape)
    print(y_train.shape)
    # fit_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=25, batch_size=32)
    fit_history = model.fit(x=x_train, y=y_train, validation_split=0.7, epochs=25, verbose=2, batch_size=32)

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(fit_history.history['accuracy'])
    plt.plot(fit_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

    model.save("myModel")




