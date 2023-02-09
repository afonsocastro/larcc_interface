#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
from config.definitions import ROOT_DIR
from sklearn.metrics import confusion_matrix
import numpy as np
from neural_networks.utils import NumpyArrayEncoder, prediction_classification
import json


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
    # feedforward.utils.plot_model(model, show_shapes=True, rankdir="LR")
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


if __name__ == '__main__':

    # n_times = 2
    n_times = 100

    output_neurons = 4
    validation_split = 0.3

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    model_config = json.load(open('model_config_optimized_' + str(output_neurons) + '_outputs.json'))

    training_test_list = []

    # printProgressBar(0, n_times, prefix='Progress:', suffix='Complete', length=80)

    for n in range(0, n_times):

        model = create_model_from_json(input_shape=650, model_config=model_config, output_shape=output_neurons)

        # sorted_data_for_learning = SortedDataForLearning(
        #     path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

        sorted_data_for_learning = SortedDataForLearning(path=ROOT_DIR + "/data_storage/data/processed_learning_data/")

        training_data = sorted_data_for_learning.training_data

        test_data = sorted_data_for_learning.test_data

        validation_n = len(training_data) * validation_split

        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config["early_stop_patience"])

        print("\n")
        print("------------------------------------------------------------------------------------------------------")
        print("TRAINING %d time" % n)
        print("------------------------------------------------------------------------------------------------------")
        print("\n")

        print("\n")
        print("Using %d samples for training and %d for validation" % (len(training_data) - validation_n, validation_n))
        print("\n")

        fit_history = model.fit(x=training_data[:, :-1], y=training_data[:, -1], validation_split=validation_split,
                                batch_size=model_config["batch_size"],
                                shuffle=True, epochs=model_config["epochs"], verbose=0, callbacks=[callback])
                                # shuffle=True, epochs=model_config["epochs"], verbose=2, callbacks=[callback])

        # printProgressBar(n/2, n_times, prefix='Progress:', suffix='Complete', length=80)

        print("\n")
        print("------------------------------------------------------------------------------------------------------")
        print("TESTING %d time" % n)
        print("------------------------------------------------------------------------------------------------------")
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
            prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=0)
            # prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=2)

            decoded_prediction = np.argmax(prediction)
            true = test_data[i, -1]

            # print("prediction")
            # print(prediction)
            # print("prediction.shape")
            # print(prediction.shape)
            # print("type(prediction)")
            # print(type(prediction))

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
        # printProgressBar(n, n_times, prefix='Progress:', suffix='Complete', length=80)

    with open("training_testing_n_times_2nd_round/training_testing_n_times.json", "w") as write_file:
        json.dump(training_test_list, write_file, cls=NumpyArrayEncoder)
