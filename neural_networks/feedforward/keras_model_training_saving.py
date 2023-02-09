#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import json
from config.definitions import ROOT_DIR


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
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    # feedforward.utils.plot_model(model, show_shapes=True)

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
    output_neurons = 4
    validation_split = 0.3

    model_config = json.load(open('model_config_optimized_' + str(output_neurons) + '_outputs.json'))

    model = create_model_from_json(input_shape=650, model_config=model_config, output_shape=output_neurons)

    # sorted_data_for_learning = SortedDataForLearning(path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    sorted_data_for_learning = SortedDataForLearning(path=ROOT_DIR + "/data_storage/data/processed_learning_data/")

    training_data = sorted_data_for_learning.training_data

    validation_n = len(training_data) * validation_split

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config["early_stop_patience"])

    fit_history = model.fit(x=training_data[:, :-1], y=training_data[:, -1], validation_split=validation_split,
                            batch_size=model_config["batch_size"],
                            shuffle=True, epochs=model_config["epochs"], verbose=2, callbacks=[callback])
                            # shuffle=True, epochs=1000, verbose=2)

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

    print("\n")
    print("Using %d samples for training and %d for validation" % (len(training_data) - validation_n, validation_n))
    print("\n")

    model.save("myModel")

