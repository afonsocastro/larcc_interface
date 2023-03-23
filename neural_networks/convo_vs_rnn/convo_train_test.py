#!/usr/bin/env python3

from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout  # create model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from config.definitions import ROOT_DIR
import numpy as np
from neural_networks.utils import NumpyArrayEncoder, prediction_classification, printProgressBar
import json
from progressbar import progressbar
import os


def create_convolutional_nn(layers):
    # Create model
    modelo = Sequential()

    for layer in layers:
        if layer["n"] == 1:
            modelo.add(Conv2D(layer["filters"], kernel_size=layer["kernel"], activation=layer["activation"],
                              input_shape=(50, 13, 1)))
            if layer["pooling"]:
                modelo.add(MaxPooling2D((2, 1)))
        else:
            modelo.add(Conv2D(layer["filters"], kernel_size=layer["kernel"], activation=layer["activation"]))
            if layer["pooling"]:
                modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))

    # compile model using accuracy to measure model performance
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo


if __name__ == '__main__':
    # n_times = 100
    n_times = 1
    # n_times = 5
    validation_split = 0.3
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    final_dict = []
    m = [{"n": 1, "kernel": (50, 1), "strides": (1, 1), "filters": 64, "activation": "relu", "pooling": False},
         {"n": 2, "kernel": (1, 1), "strides": (1, 1), "filters": 32, "activation": "relu", "pooling": False}]

    model = create_convolutional_nn(m)
    print(model.summary())

    path = ROOT_DIR + "/data_storage/data/universal_norm/user_splitted_data/"

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    training_config = json.load(f)
    f.close()

    training_data = np.empty((0, 651))
    test_data = np.empty((0, 651))

    files = os.listdir(path)
    for file in files:
        new_array = np.load(path + file)
        for user in training_config["training_users"]:
            number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
            if user == number:
                print("training data: " + str(user))
                print("file: " + file)
                training_data = np.append(training_data, new_array, axis=0)

        for user in training_config["test_users"]:
            number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
            if user == number:
                print("test data: " + str(user))
                print("file: " + file)
                test_data = np.append(test_data, new_array, axis=0)

    training_test_list = []

    for n in progressbar(range(n_times), redirect_stdout=True):

        np.random.shuffle(training_data)
        np.random.shuffle(test_data)

        n_train = len(training_data) * (1 - validation_split)
        n_val = len(training_data) * validation_split
        n_test = test_data.shape[0]
        x_train = training_data[:, :-1]
        y_train = training_data[:, -1]
        x_train = np.reshape(x_train, (training_data.shape[0], 50, 13, 1))
        y_train = to_categorical(y_train)

        # train the model
        fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, epochs=50, verbose=2,
                                batch_size=64, shuffle=True)

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
        # plt.show()

        plt.savefig(ROOT_DIR + "/neural_networks/convo_vs_rnn/training_curves.png", bbox_inches='tight')
        model.save("convolutional_universal_norm_model")
    #
    #     print("\n")
    #     print("-------------------------------------------------------------------------------------------------")
    #     print("TRAINING %d time" % n)
    #     print("-------------------------------------------------------------------------------------------------")
    #     print("\n")
    #
    #     print("\n")
    #     print("Using %d samples for training and %d for validation" % (n_train, n_val))
    #     print("\n")
    #
    #     print("\n")
    #     print("-------------------------------------------------------------------------------------------------")
    #     print("TESTING %d time" % n)
    #     print("-------------------------------------------------------------------------------------------------")
    #     print("\n")
    #
    #     predictions_list = []
    #
    #     pull = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
    #             "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    #     push = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
    #             "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    #     shake = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
    #              "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    #     twist = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
    #              "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    #
    #     for i in range(0, len(test_data)):
    #         x_test = np.reshape(test_data[i:i + 1, :-1], (1, 50, 13, 1))
    #         prediction = model.predict(x=x_test, verbose=0)
    #         # Reverse to_categorical from keras utils
    #         decoded_prediction = np.argmax(prediction, axis=1, out=None)
    #
    #         true = test_data[i, -1]
    #
    #         prediction_classification(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull,
    #                                   pred=prediction)
    #         prediction_classification(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push,
    #                                   pred=prediction)
    #         prediction_classification(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake,
    #                                   pred=prediction)
    #         prediction_classification(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist,
    #                                   pred=prediction)
    #
    #         predictions_list.append(decoded_prediction)
    #
    #     predicted_values = np.asarray(predictions_list)
    #     cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)
    #
    #     cm_true = cm / cm.astype(float).sum(axis=1)
    #
    #     test_dict = {"cm_true": cm_true, "cm": cm, "pull": pull, "push": push, "shake": shake, "twist": twist}
    #     training_test_dict = {"training": fit_history.history, "test": test_dict}
    #     training_test_list.append(training_test_dict)
    #
    # final_dict = {"train_test_list": training_test_list}
    #
    # with open("n_times_train_test/n_times_train_test.json", "w") as write_file:
    #     json.dump(final_dict, write_file, cls=NumpyArrayEncoder)
