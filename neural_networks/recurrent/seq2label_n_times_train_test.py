#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
from neural_networks.utils import NumpyArrayEncoder, prediction_classification
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
from neural_networks.utils import plot_confusion_matrix_percentage
import json
from progressbar import progressbar

if __name__ == '__main__':
    n_times = 100

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    params = 12
    time_steps = 50
    batch_size = 64
    epochs = 50
    validation_split = 0.3
    training_test_list = []
    for n in progressbar(range(n_times), redirect_stdout=True):

        sorted_data_for_learning = SortedDataForLearning(
            path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

        training_data = sorted_data_for_learning.training_data
        test_data = sorted_data_for_learning.test_data

        n_train = len(training_data) * (1 - validation_split)
        n_val = len(training_data) * validation_split
        n_test = test_data.shape[0]

        x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], 50, 13))
        y_train = to_categorical(training_data[:, -1])

        x_train = x_train[:, :, 1:]

        model = Sequential()
        model.add(LSTM(16, input_shape=(time_steps, params), return_sequences=True))
        # model.add(LSTM(64, input_shape=(50, 13)))
        # Dropout(0.2)
        model.add(LSTM(16))
        model.add(Dense(4, activation="softmax"))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

        model.summary()
        plot_model(model, to_file="seq2label/model.png", show_shapes=True)

        callback = EarlyStopping(monitor='val_loss', patience=10)

        fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                                shuffle=True, callbacks=[callback], verbose=2)

        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print("TRAINING %d time" % n)
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        print("\n")
        print("Using %d samples for training and %d for validation" % (n_train, n_val))
        print("\n")

        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print("TESTING %d time" % n)
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        predictions_list = []

        pull = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
        push = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
        shake = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                 "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
        twist = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                 "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}

        x_test = np.reshape(test_data[:, :-1], (n_test, 50, 13))
        y_test = to_categorical(test_data[:, -1])
        x_test = x_test[:, :, 1:]

        for i in range(0, len(test_data)):
            prediction = model.predict(x=x_test[i:i + 1, :, :], verbose=0)
            decoded_prediction = np.argmax(prediction)

            true = test_data[i, -1]

            prediction_classification(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull,
                                      pred=prediction)
            prediction_classification(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push,
                                      pred=prediction)
            prediction_classification(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake,
                                      pred=prediction)
            prediction_classification(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist,
                                      pred=prediction)

            predictions_list.append(decoded_prediction)

        label_pred = model.predict(x=x_test, verbose=0)
        results = np.argmax(label_pred, axis=1, out=None)
        y_results = np.argmax(y_test, axis=1, out=None)

        cm = confusion_matrix(y_true=y_results, y_pred=results)
        cm_true = cm / cm.astype(float).sum(axis=1)

        test_dict = {"cm_true": cm_true, "cm": cm, "pull": pull, "push": push, "shake": shake, "twist": twist}
        training_test_dict = {"training": fit_history.history, "test": test_dict}
        training_test_list.append(training_test_dict)

    with open(ROOT_DIR + "/neural_networks/recurrent/seq2label_n_times/training_testing_n_times_seq2label.json", "w") as wf:
        json.dump(training_test_list, wf, cls=NumpyArrayEncoder)
