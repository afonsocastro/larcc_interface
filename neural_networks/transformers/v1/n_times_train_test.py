#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs: INFO, WARN, and DEBUG
import sys
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from sklearn.metrics import confusion_matrix
from larcc_interface.neural_networks.utils import plot_confusion_matrix_percentage, prediction_classification, \
    simple_metrics_calc, \
    prediction_classification_absolute
from larcc_interface.config.definitions import ROOT_DIR
import numpy as np
import json
from progressbar import progressbar
from larcc_interface.neural_networks.utils import NumpyArrayEncoder
from larcc_interface.neural_networks.transformers.v1.create_models import (create_transformer_v1_0,
                                                                           create_transformer_v1_1,
                                                           create_transformer_v1_2, create_transformer_v1_3,
                                                           create_transformer_v1_4, create_transformer_v1_5)
from rich.progress import track

if __name__ == '__main__':
    n_times = 100
    validation_split = 0.3
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    #models = ["model_v1_1", "model_v1_2", "model_v1_3", "model_v1_4", "model_v1_5"]
    #for transformer_model_olarilole in models:
    training_test_list = []
    for n in progressbar(range(n_times), redirect_stdout=True):
    # for n in track(range(n_times), total=n_times):
        training_data = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_train_data_20ms.npy")
        x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], 20, 13))
        y_train = to_categorical(training_data[:, -1])
        x_train = x_train[:, :, 1:]

        model, model_name = create_transformer_v1_5()

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        fit_history = model.fit(x_train, y_train, shuffle=True, validation_split=validation_split, epochs=500,
                                batch_size=32)

        print("\n")
        print("Using %d samples for training and %d for validation" % (
        len(training_data) * (1 - validation_split), len(training_data) * validation_split))

        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print("TESTING: %d simulation" % n)
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        test_data_1 = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_test_data_20ms.npy")

        x_test = test_data_1[:, :-1]
        y_test = test_data_1[:, -1]
        x_test = np.reshape(x_test, (test_data_1.shape[0], 20, 13, 1))

        x_test = x_test[:, :, 1:, :]

        predictions_list = []

        pull = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        push = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        shake = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
        twist = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

        for i in range(0, len(test_data_1)):
            prediction = model.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
            # Reverse to_categorical from keras utils
            decoded_prediction = np.argmax(prediction, axis=1, out=None)
            true = y_test[i]

            prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
            prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
            prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
            prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

            predictions_list.append(decoded_prediction)

        predicted_values = np.asarray(predictions_list)

        cm = confusion_matrix(y_true=y_test, y_pred=predicted_values)
        cm_true = cm / cm.astype(float).sum(axis=1)
        cm_true_percentage = cm_true * 100

        test_dict = {"cm_true": cm_true, "cm": cm, "pull": pull, "push": push, "shake": shake, "twist": twist}
        training_test_dict = {"training": fit_history.history, "test": test_dict}
        training_test_list.append(training_test_dict)

    with open(str(n_times) + "_times_train_test_" + model_name + ".json", "w") as write_file:
        json.dump(training_test_list, write_file, cls=NumpyArrayEncoder)
