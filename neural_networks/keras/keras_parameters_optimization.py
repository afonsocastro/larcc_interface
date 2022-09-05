#!/usr/bin/env python3

from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
import numpy as np
import time
from tensorflow import keras
from tabulate import tabulate
import itertools
import json
from statistics import mean

# batch_size_options = [32, 64, 96, 192, 256]
batch_size_options = [96]
epochs_options = [500]
n_layers_options = [1, 2, 3]
neurons_per_layer_option = [16, 32, 64]
learning_rate_options = [0.01, 0.001, 0.0001]
dropout_options = [0, 0.2, 0.5]
# activation_options = ['relu', 'sigmoid', 'softsign', 'tanh', 'selu']
activation_options = ['relu', 'softsign', 'tanh', 'selu']
# optimizer_options = ["Adam", "SGD", "RMSprop"]
optimizer_options = ["Adam"]
loss_options = ['sparse_categorical_crossentropy']

# model_options = [neurons_per_layer_option, dropout_options, activation_options]
model_options = [neurons_per_layer_option, activation_options]
layer_options = list(itertools.product(*model_options))

early_stop_patience = 20
best_results_printed = 10


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=3, length=100, fill='█', printEnd="\r"):
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def writing_model_config_json(lr, optimizer, dropo, loss, batch_size, epochs, model_combination):

    layers = []
    n_layers = len(model_combination)
    for layer in range(0, n_layers):
        neurons = model_combination[layer][0]
        # dropout = model_combination[layer][1]
        activation = model_combination[layer][1]
        # activation = model_combination[layer][2]
        # layer = {"id": layer, "neurons": neurons, "dropout": dropout, "activation": activation}
        layer = {"id": layer, "neurons": neurons, "activation": activation}
        layers.append(layer)

    data = {'layers': layers, 'lr': lr, 'n_layers': n_layers, 'dropout': dropo, 'optimizer': optimizer, 'loss': loss,
            'batch_size': batch_size, 'epochs': epochs, 'early_stop_patience': early_stop_patience}

    with open('model_config.json', 'w') as fp:
        json.dump(data, fp)


def build_model(layer_combination_list, dro, learning_rate, _optimizer, _loss):
# def build_model(layer_combination_list, _optimizer, _loss):

    n_layers = len(layer_combination_list)
    model = Sequential()

    for layer in range(0, n_layers):

        neurons = layer_combination_list[layer][0]
        # dropout = layer_combination_list[layer][1]
        activation = layer_combination_list[layer][1]
        # activation = layer_combination_list[layer][2]

        if layer == 0:
            model.add(Dense(units=neurons, input_shape=(650,), activation=activation, kernel_regularizer='l1'))
            # model.add(Dropout(dropout))
        elif layer == n_layers - 1:
            model.add(Dense(units=neurons, activation=activation, kernel_regularizer='l1'))
            model.add(Dropout(dro))
        else:
            model.add(Dense(units=neurons, activation=activation, kernel_regularizer='l1'))
            # model.add(Dropout(dropout))

    model.add(Dense(units=4, activation='softmax'))

    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=10000,
    #     decay_rate=0.9)

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

    total_tests = 0
    for batch_size in batch_size_options:
        for epoch in epochs_options:
            for lr in learning_rate_options:
                for optimizer in optimizer_options:
                    for loss in loss_options:
                        for do in dropout_options:
                            for layers in n_layers_options:
                                model_combinations = list(itertools.product(layer_options, repeat=layers))
                                for model_combination in model_combinations:
                                    # total_tests += 1
                                    total_tests += 3

    print("total_tests")
    print(total_tests)

    print("total_configurations")
    print(int(total_tests/3))

    total_time = 0
    results_list = []
    val_accuracies = []
    n_test = 0

    printProgressBar(0, total_tests, prefix='Progress:', suffix='Complete', length=150)

    for batch_size in batch_size_options:
        for epoch in epochs_options:
            for lr in learning_rate_options:
                for optimizer in optimizer_options:
                    for loss in loss_options:
                        for do in dropout_options:
                            for layers in n_layers_options:
                                model_combinations = list(itertools.product(layer_options, repeat=layers))
                                for model_combination in model_combinations:

                                    val_accuracies_for_mean = []
                                    last_epochs_for_mean = []

                                    star_time = time.time()

                                    model = build_model(model_combination, do, lr, optimizer, loss)
                                    # model = build_model(model_combination, optimizer, loss)

                                    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                             patience=early_stop_patience)
                                    for trial in range(0, 3):
                                        fit_history = model.fit(x=all_data[:, :-1], y=all_data[:, -1],
                                                                validation_split=validation_split,
                                                                batch_size=batch_size, shuffle=True, epochs=epoch,
                                                                verbose=0, callbacks=[callback])
                                        val_accuracies_for_mean.append(fit_history.history['val_accuracy'][-1])
                                        last_epochs_for_mean.append(len(fit_history.history['loss']))

                                        time.sleep(0.1)
                                        # Update Progress Bar
                                        printProgressBar(n_test, total_tests, prefix='Progress:', suffix='Complete',
                                                         length=150)
                                        n_test += 1

                                    end_time = time.time()
                                    elapsed_time = end_time - star_time

                                    val_accuracy_mean = round(mean(val_accuracies_for_mean), 6)
                                    last_epoch_mean = round(mean(last_epochs_for_mean), 2)

                                    test = {"n_test": n_test, "val_accuracy": val_accuracy_mean,
                                            "batch_size": batch_size, "epochs": epoch, "layers": layers, "lr": lr,
                                            "dropout": do, "model": model_combination, "optimizer": optimizer,
                                            "last_epoch": last_epoch_mean, "loss": loss}

                                    val_accuracies.append(val_accuracy_mean)
                                    results_list.append(test)

                                    total_time = total_time + elapsed_time

    print("\n")
    print("\n")

    # Just for printing beautiful tables with values------------------------------
    results_for_printing = []
    headers = []

    ordered_results_list = sorted(results_list, key=lambda x: x['val_accuracy'], reverse=True)

    for i in range(0, int(best_results_printed)):
        sample_result = []
        headers = []
        for key, value in ordered_results_list[i].items():
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

    writing_model_config_json(best_parameters["lr"], best_parameters["optimizer"], best_parameters["dropout"], best_parameters["loss"],
                              best_parameters["batch_size"], best_parameters["epochs"], best_parameters["model"])
