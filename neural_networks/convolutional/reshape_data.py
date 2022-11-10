#!/usr/bin/env python3

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten  # create model

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np

if __name__ == '__main__':
    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data

    print("training_data.shape")
    print(training_data.shape)
    print("type(training_data)")
    print(type(training_data))
    print("training_data[0, 0:27]")
    print(training_data[0, 0:27])
    print("training_data[0, 0:27].shape")
    print(training_data[0, 0:27].shape)

    for i in range(0, training_data.shape[0]):
        if i < 1425:
            sample = training_data[i, :-1].reshape(13, 50).reshape(-1)

        # elif i >= 1425:
        #     print(sample)

    print("sample.shape")
    print(sample.shape)
    print("sample")
    print(sample)
    # print("training_data[0:3, 0:1]")
    # print(training_data[0, 0:1])
    # x_train = training_data[:1425, :-1].reshape(1425, 50, 13, 1)
    # x_val = training_data[1425:, :-1].reshape(611, 13, 50, 1)


    # print("x_train[0, 0:7]")
    # print("x_train[0].shape")
    # print(x_train[0].shape)
    # print("x_train[0][0, 0:7]")
    # print(x_train[0][0, 0:7])
    # print("x_val.shape")
    # print(x_val.shape)
    #
    #

    # reshape data to fit model
    # X_train = training_data.reshape(60000, 28, 28, 1)
    # X_test = X_test.reshape(10000, 28, 28, 1)

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # print(y_train[0])
    # model = Sequential()  # add model layers
    # model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    # model.add(Conv2D(32, kernel_size=3, activation="relu"))
    # model.add(Flatten())
    # model.add(Dense(10, activation="softmax"))
    #
    # # compile model using accuracy to measure model performance
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # # train the model
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
