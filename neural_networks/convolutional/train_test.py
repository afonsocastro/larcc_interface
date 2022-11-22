#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
from neural_networks.utils import plot_confusion_matrix_percentage

if __name__ == '__main__':
    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data
    validation_split = 0.7
    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1-validation_split)
    n_test = test_data.shape[0]
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    x_train = np.reshape(x_train, (training_data.shape[0], 50, 13, 1))
    y_train = to_categorical(y_train)

    # Create model
    model = Sequential()
    # model.add(Conv2D(64, kernel_size=(50, 1), activation="relu", input_shape=(50, 13, 1)))
    model.add(Conv2D(64, kernel_size=(13, 1), activation="relu", input_shape=(50, 13, 1)))
    model.add(MaxPooling2D((2, 1)))

    model.add(Conv2D(32, kernel_size=(4, 1), activation="relu"))
    model.add(MaxPooling2D((2, 1)))

    # model.add(Dropout(0.1))
    model.add(Flatten())
    # model.add(Dropout(0.1))
    model.add(Dense(4, activation="softmax"))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # train the model
    fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, epochs=50, verbose=2, batch_size=64,
                            shuffle=True)

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

    print("\n")
    print("Using %d samples for training and %d for validation" % (n_train, n_val))
    print("\n")

    model.save("myModel")

    x_test = np.reshape(test_data[:, :-1], (n_test, 50, 13, 1))
    predicted_values = model.predict(x=x_test, verbose=2)

    # Reverse to_categorical from keras utils
    predicted_values = np.argmax(predicted_values, axis=1, out=None)

    true_values = test_data[:, -1]

    cm = confusion_matrix(y_true=true_values, y_pred=predicted_values)

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/convolutional/predicted_data/confusion_matrix.png", bbox_inches='tight')

    cm_true = cm / cm.astype(float).sum(axis=1)
    cm_true_percentage = cm_true * 100
    plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
                                     title="Confusion Matrix (%) - CONVOLUTIONAL")
    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/convolutional/predicted_data/confusion_matrix_true.png", bbox_inches='tight')
