#!/usr/bin/env python3

from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
from tensorflow import keras
from tabulate import tabulate
import itertools
import json
from statistics import mean
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning

# batch_size_options = [32, 64, 96, 192, 256]
batch_size_options = [96]
epochs_options = [300]
n_layers_options = [3]
neurons_per_layer_option = [32, 64]
learning_rate_options = [0.001]
dropout_options = [0.2]
# activation_options = ['relu', 'sigmoid', 'softsign', 'tanh', 'selu']
activation_options = ['relu', 'selu']
# optimizer_options = ["Adam", "SGD", "RMSprop"]
optimizer_options = ["Adam"]
loss_options = ['sparse_categorical_crossentropy']

# model_options = [neurons_per_layer_option, dropout_options, activation_options]
model_options = [neurons_per_layer_option, activation_options]
layer_options = list(itertools.product(*model_options))

early_stop_patience = 30
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


def writing_model_config_json(lr, optimizer, dropo, loss, batch_size, epochs, model_combination, o_n):

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

    with open('model_config_optimized_' + str(o_n) + '_outputs.json', 'w') as fp:
        json.dump(data, fp)


def build_model(layer_combination_list, dro, learning_rate, _optimizer, _loss, output_shape):

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

    model.add(Dense(units=output_shape, activation='softmax'))

    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    # lr_schedule = feedforward.optimizers.schedules.ExponentialDecay(
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
    output_neurons = 4
    median_n_tests = 1

    validation_split = 0.3

    sorted_data_for_learning = SortedDataForLearning()

    training_data = sorted_data_for_learning.trainning_data

    validation_n = len(training_data) * validation_split

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
                                    total_tests += median_n_tests

    print("\ntotal_tests")
    print(total_tests)

    print("\nMedian of %d tests evaluation, for each network configuration!\n" %(median_n_tests))

    print("total_configurations")
    print(int(total_tests/median_n_tests))

    print("\n")

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

                                    model = build_model(model_combination, do, lr, optimizer, loss, output_shape=output_neurons)

                                    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                             patience=early_stop_patience)
                                    for trial in range(0, median_n_tests):
                                        fit_history = model.fit(x=training_data[:, :-1], y=training_data[:, -1],
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
                              best_parameters["batch_size"], best_parameters["epochs"], best_parameters["model"], output_neurons)
