#!/usr/bin/env python3

from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.activations import sigmoid, elu, relu, softmax
import matplotlib.pyplot as plt
import numpy as np
import time
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from tabulate import tabulate


# batch_size_options = [32, 64, 96, 192, 256]
batch_size_options = [64, 96]
epochs_options = [300]
# n_layers_options = [1, 2, 3, 4]
n_layers_options = [2, 3]
# neurons_per_layer_option = [16, 32, 64, 128]
neurons_per_layer_option = [64, 128]
# learning_rate_options = [0.0001, 0.001, 0.01]
learning_rate_options = [0.001]
# dropout_options = [0.1, 0.2, 0.5]
dropout_options = [0.1]
# activation_options = ['relu', 'sigmoid', 'softsign', 'tanh', 'selu', 'softmax']
activation_options = ['relu', 'sigmoid']
# optimizer_options = ["Adam", "SGD", "Nadam", "RMSprop"]
optimizer_options = ["Adam", "SGD"]
loss_options = ['sparse_categorical_crossentropy']


early_stop_patience = 20


# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def build_model(n_layers, neurons_per_layer, learning_rate, dropout, _optimizer, activation, _loss):

    model = Sequential()
    for layer in range(0, n_layers):
        if layer == 0:
            model.add(Dense(units=neurons_per_layer, input_shape=(650,), activation=activation))
            model.add(Dropout(dropout))
        else:
            model.add(Dense(units=neurons_per_layer, activation=activation))
            model.add(Dropout(dropout))

    model.add(Dense(units=4, activation='softmax'))

    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    # model.summary()
    if _optimizer == "Adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=_loss,
                      metrics=['accuracy'])
    elif _optimizer == "SGD":
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss=_loss,
                      metrics=['accuracy'])
    elif _optimizer == "Nadam":
        model.compile(optimizer=Nadam(learning_rate=learning_rate), loss=_loss,
                      metrics=['accuracy'])
    elif _optimizer == "RMSprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=_loss,
                      metrics=['accuracy'])

    return model


if __name__ == '__main__':

    validation_split = 0.3

    all_data = np.load('../data/learning_data_training.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    validation_n = len(all_data) * validation_split

    all_data = np.array(all_data)

    # tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=1)
    # tuner.search(x=x_train, y=y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val))
    # best_model = tuner.get_best_models()[0]

    # almost_every_tests = len(batch_size_options)

    total_tests = len(batch_size_options) * len(epochs_options) * len(n_layers_options) * \
                  len(neurons_per_layer_option) * len(learning_rate_options) * len(dropout_options) * \
                  len(optimizer_options) * len(activation_options) * len(loss_options)


    total_time = 0
    results_list = []
    val_accuracies = []
    n_test = 0

    printProgressBar(0, total_tests, prefix='Progress:', suffix='Complete', length=150)

    for batch_size in batch_size_options:
        for epoch in epochs_options:
            for layers in n_layers_options:
                for neurons in neurons_per_layer_option:
                    for lr in learning_rate_options:
                        for do in dropout_options:
                            for optimizer in optimizer_options:
                                for activ in activation_options:
                                    for loss in loss_options:

                                        star_time = time.time()

                                        model = build_model(layers, neurons, lr, do, optimizer, activ, loss)

                                        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=early_stop_patience)

                                        fit_history = model.fit(x=all_data[:, :-1], y=all_data[:, -1],
                                                                validation_split=validation_split,
                                                                batch_size=batch_size, shuffle=True, epochs=epoch,
                                                                verbose=0, callbacks=[callback])

                                        end_time = time.time()
                                        elapsed_time = end_time - star_time

                                        test = {"n_test": n_test, "val_accuracy": fit_history.history['val_accuracy'][-1],
                                                "batch_size": batch_size, "epochs": epoch, "layers": layers,
                                                "neurons": neurons, "learning_rate": lr, "dropout": do,
                                                "optimizer": optimizer, "activation": activ, "loss": loss,
                                                "last_epoch": len(fit_history.history['loss'])}

                                        val_accuracies.append(fit_history.history['val_accuracy'][-1])
                                        results_list.append(test)

                                        total_time = total_time + elapsed_time
                                        time.sleep(0.1)
                                        # Update Progress Bar
                                        printProgressBar(n_test, total_tests, prefix='Progress:', suffix='Complete', length=150)
                                        n_test += 1

    print("\n")
    print("\n")

    # Just for printing beautiful tables with values------------------------------
    results_for_printing = []
    headers = []

    for i in range(0, n_test):
        sample_result = []
        headers = []
        for key, value in results_list[i].items():
            sample_result.append(value)
            headers.append(key)
        results_for_printing.append(sample_result)
    print(tabulate(results_for_printing, headers=headers, tablefmt="fancy_grid"))
    print("\n")
    # -----------------------------------------------------------------------------------------

    max_accuracy = max(val_accuracies)
    max_indice = val_accuracies.index(max_accuracy)
    best_parameters = results_list[max_indice]

    print("\nbest_parameters")
    values = []
    keys = []
    for key, value in best_parameters.items():
        values.append(value)
        keys.append(key)
    print(tabulate([values], headers=keys, tablefmt="fancy_grid"))
    print("\n")

    print("\ntotal_time: %f" % total_time)

