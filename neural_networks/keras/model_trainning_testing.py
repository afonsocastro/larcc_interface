#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import json
from config.definitions import ROOT_DIR
from itertools import product
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
from tabulate import tabulate
from colorama import Fore


def create_model_from_json(input_shape, model_config, output_shape):
    model = Sequential()

    for layer in model_config["layers"]:
        if layer["id"] == 0:
            model.add(Dense(units=layer["neurons"], input_shape=(input_shape,), activation=layer["activation"],
                            kernel_regularizer='l1'))
            # model.add(Dropout(layer["dropout"]))
        elif layer["id"] == model_config["n_layers"] - 1:
            model.add(Dense(units=layer["neurons"], activation=layer["activation"], kernel_regularizer='l1'))
            model.add(Dropout(model_config["dropout"]))
        else:
            model.add(Dense(units=layer["neurons"], activation=layer["activation"], kernel_regularizer='l1'))
            # model.add(Dropout(layer["dropout"]))

    model.add(Dense(units=output_shape, activation='softmax'))

    # `rankdir='LR'` is to make the graph horizontal.
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    keras.utils.plot_model(model, show_shapes=True)

    model.summary()

    if model_config["optimizer"] == "Adam":
        model.compile(optimizer=Adam(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])
    elif model_config["optimizer"] == "SGD":
        model.compile(optimizer=SGD(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])
    elif model_config["optimizer"] == "Nadam":
        model.compile(optimizer=Nadam(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])
    elif model_config["optimizer"] == "RMSprop":
        model.compile(optimizer=RMSprop(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])

    return model


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {iteration} out of {total} Training & Test | ({percent}% {suffix})', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def prediction_classification(cla, true_out, dec_pred, dictionary, pred):
    if true_out == cla and dec_pred == cla:
        dictionary["true_positive"] = np.append(dictionary["true_positive"], pred, axis=0)
    elif true_out != cla and dec_pred == cla:
        dictionary["false_positive"] = np.append(dictionary["false_positive"], pred, axis=0)
    elif true_out == cla and dec_pred != cla:
        dictionary["false_negative"] = np.append(dictionary["false_negative"], pred, axis=0)


if __name__ == '__main__':

    n_times = 4
    # n_times = 100
    output_neurons = 4
    validation_split = 0.3

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    model_config = json.load(open('model_config_optimized_' + str(output_neurons) + '_outputs.json'))

    model = create_model_from_json(input_shape=650, model_config=model_config, output_shape=output_neurons)

    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.trainning_data

    test_data = sorted_data_for_learning.test_data

    validation_n = len(training_data) * validation_split

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config["early_stop_patience"])

    training_test_list = []

    printProgressBar(0, n_times, prefix='Progress:', suffix='Complete', length=120)

    for n in range(0, n_times):

        # print("\n")
        # print("------------------------------------------------------------------------------------------------------")
        # print("TRAINING %d time" % n)
        # print("------------------------------------------------------------------------------------------------------")
        # print("\n")
        #
        # print("\n")
        # print("Using %d samples for training and %d for validation" % (len(training_data) - validation_n, validation_n))
        # print("\n")

        fit_history = model.fit(x=training_data[:, :-1], y=training_data[:, -1], validation_split=validation_split,
                                batch_size=model_config["batch_size"],
                                shuffle=True, epochs=model_config["epochs"], verbose=0, callbacks=[callback])
                                # shuffle=True, epochs=model_config["epochs"], verbose=2, callbacks=[callback])

        printProgressBar(n/2, n_times, prefix='Progress:', suffix='Complete', length=120)

        # print("\n")
        # print("------------------------------------------------------------------------------------------------------")
        # print("TESTING %d time" % n)
        # print("------------------------------------------------------------------------------------------------------")
        # print("\n")

        predictions_list = []

        col = test_data.shape[1]

        pull = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                "false_negative": np.empty((0, n_labels))}
        push = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                "false_negative": np.empty((0, n_labels))}
        shake = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                 "false_negative": np.empty((0, n_labels))}
        twist = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                 "false_negative": np.empty((0, n_labels))}

        for i in range(0, len(test_data)):
            prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=0)
            # prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=2)

            predict_lst = []
            for label in range(0, n_labels):
                predict_lst.append(prediction[0][label])

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

        predicted_values = np.asarray(predictions_list)
        cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)
        cm_true = cm / cm.astype(float).sum(axis=1)

        test_dict = {"cm_true": cm_true, "pull": pull, "push": push, "shake": shake, "twist": twist}
        training_test_dict = {"training": fit_history, "test": test_dict}
        training_test_list.append(training_test_dict)
        printProgressBar(n, n_times, prefix='Progress:', suffix='Complete', length=120)

    import csv

    print("\nWrite a list of dictionaries to a CSV file:")
    fw = open("test1.csv", "w", newline='')
    writer = csv.DictWriter(fw, fieldnames=["training", "test"])
    writer.writeheader()
    writer.writerows(training_test_list)
    fw.close()

    cm_cumulative = np.empty((n_labels, n_labels))
    for n_test in range(0, len(training_test_list)):
        # CONFUSION MATRIX
        print("cm_true")
        print(training_test_list[n_test]["test"]["cm_true"])
        for i in range(0, training_test_list[n_test]["test"]["cm_true"].shape[0]):
            for j in range(0, training_test_list[n_test]["test"]["cm_true"].shape[1]):
                print(i, j)
                cm_cumulative[i][j] = cm_cumulative[i][j] + training_test_list[n_test]["test"]["cm_true"][i][j]

        print("cm_cumulative")
        print(cm_cumulative)

    cm_mean = np.empty((n_labels, n_labels))
    for i in range(0, cm_cumulative.shape[0]):
        for j in range(0, cm_cumulative.shape[1]):
            cm_mean[i][j] = cm_cumulative[i][j] / n_times

    print("cm_mean")
    print(cm_mean)
