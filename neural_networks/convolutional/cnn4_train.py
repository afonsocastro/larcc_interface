#!/usr/bin/env python3
#
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout  # create model
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np


def create_convolutional_nn(input):
    # Create model CNN 4
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(5, 1), activation="relu", input_shape=(input, 13, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))

    # compile model using accuracy to measure model performance
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo


if __name__ == '__main__':

    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"

    # time_windows = 5
    time_window = 2
    n_times = 100
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    # for time_window in progressbar(range(1, time_windows+1), redirect_stdout=True):

    config_file = "training_config_time_" + str(time_window)
    training_test_list = []
    # for n in progressbar(range(n_times), redirect_stdout=True):

    sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    print("\n")
    print("config_file")
    print(config_file)
    print("\n")

    input_nn = time_window * 10
    validation_split = 0.3
    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    n_test = test_data.shape[0]
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    x_train = np.reshape(x_train, (training_data.shape[0], input_nn, 13, 1))
    y_train = to_categorical(y_train)

    model = create_convolutional_nn(input_nn)

    # train the model
    fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, epochs=50, verbose=2,
                            batch_size=64, shuffle=True)

    model.save("cnn4_model_02_time_window")

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
    #
    plt.show()