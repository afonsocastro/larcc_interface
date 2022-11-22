#!/usr/bin/env python3
#
from keras.utils import to_categorical  # one-hot encode target column
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout  # create model
from sklearn.metrics import confusion_matrix
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
from neural_networks.utils import NumpyArrayEncoder, prediction_classification, printProgressBar
import json
from progressbar import progressbar


def create_convolutional_nn(input):
    # Create model
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

    time_windows = 5
    n_times = 100
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    for time_window in progressbar(range(1, time_windows+1), redirect_stdout=True):

        config_file = "training_config_time_" + str(time_window)
        training_test_list = []
        for n in progressbar(range(n_times), redirect_stdout=True):

            sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
            training_data = sorted_data_for_learning.training_data
            test_data = sorted_data_for_learning.test_data

            print("\n")
            print("config_file")
            print(config_file)
            print("\n")

            input_nn = time_window * 10
            validation_split = 0.7
            n_train = len(training_data) * validation_split
            n_val = len(training_data) * (1 - validation_split)
            n_test = test_data.shape[0]
            x_train = training_data[:, :-1]
            y_train = training_data[:, -1]
            x_train = np.reshape(x_train, (training_data.shape[0], input_nn, 13, 1))
            y_train = to_categorical(y_train)

            model = create_convolutional_nn(input_nn)

            # train the model
            fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, epochs=50, verbose=2,
                                    batch_size=64, shuffle=True)

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

            for i in range(0, len(test_data)):
                x_test = np.reshape(test_data[i:i + 1, :-1], (1, input_nn, 13, 1))
                prediction = model.predict(x=x_test, verbose=0)

                # Reverse to_categorical from keras utils
                decoded_prediction = np.argmax(prediction, axis=1, out=None)

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

            predicted_values = np.asarray(predictions_list)
            cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)

            cm_true = cm / cm.astype(float).sum(axis=1)

            test_dict = {"cm_true": cm_true, "cm": cm, "pull": pull, "push": push, "shake": shake, "twist": twist}
            training_test_dict = {"training": fit_history.history, "test": test_dict}
            training_test_list.append(training_test_dict)

        final_dict = {"time_window": time_window, "train_test_list": training_test_list}

        with open("x_time_window_train_test/x_time_window_train_test_" + str(time_window) + ".json", "w") as write_file:
            json.dump(final_dict, write_file, cls=NumpyArrayEncoder)

        del model
