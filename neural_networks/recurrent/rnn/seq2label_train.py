#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage

if __name__ == '__main__':
    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150
    validation_split = 0.7
    time_window = 2
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    config_file = "training_config_time_" + str(time_window)
    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1 - validation_split)
    n_test = test_data.shape[0]

    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], time_steps, 13))
    y_train = to_categorical(training_data[:, -1])
    x_test = np.reshape(test_data[:, :-1], (n_test, time_steps, 13))
    y_test = to_categorical(test_data[:, -1])

    x_train = x_train[:, :, 1:]
    x_test = x_test[:, :, 1:]

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = Sequential()
    model.add(LSTM(16, input_shape=(time_steps, params), return_sequences=True))
    # model.add(LSTM(16, input_shape=(time_steps, params)))
    model.add(LSTM(16))
    model.add(Dense(4, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    model.summary()
    # plot_model(model, to_file="seq2label/model.png", show_shapes=True)

    callback = EarlyStopping(monitor='val_loss', patience=10)

    fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, batch_size=batch_size,
                            epochs=epochs, shuffle=True, callbacks=[callback], verbose=2)
    # model.save("seq2label_20ms")
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
    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2label/training_curves_RNN_adjusted_norm.png", bbox_inches='tight')

    # predicted_values = model.predict(x_test)
    #
    # results = np.argmax(predicted_values, axis=1, out=None)
    # y_results = np.argmax(y_test, axis=1, out=None)
    #
    # cm = confusion_matrix(y_true=y_results, y_pred=results)
    # print("cm")
    # print(cm)
    #
    # labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    # n_labels = len(labels)
    # blues = plt.cm.Blues
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(cmap=blues)
    #
    # # plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2label/confusion_matrix.png", bbox_inches='tight')
    # cm_true = cm / cm.astype(float).sum(axis=1)
    # cm_true_percentage = cm_true * 100
    # plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
    #                                  title="Confusion Matrix (%) - Seq-To-Label")
    # # plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2label/confusion_matrix_true.png", bbox_inches='tight')
