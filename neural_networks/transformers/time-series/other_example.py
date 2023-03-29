#!/usr/bin/env python3
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, n_classes, dropout=0,
                mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    time_window = 2
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    input_nn = time_window * 10
    validation_split = 0.3

    # data = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_data.npy")
    # print(data.shape)
    #
    # training_data = data[0:2036, :]
    # test_data = data[2036:2558, :]



    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"

    time_window = 2
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    config_file = "training_config_time_" + str(time_window)

    sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    print("\n")
    print("config_file")
    print(config_file)
    print("\n")


    print(test_data.shape)
    print(training_data.shape)

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    n_test = test_data.shape[0]
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    x_train = np.reshape(x_train, (training_data.shape[0], input_nn, 13))
    y_train = to_categorical(y_train)
    input_shape = x_train.shape[1:]

    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, n_classes=n_labels,
                        mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

    # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #               metrics=["sparse_categorical_accuracy"])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

    fit_history = model.fit(x_train, y_train, validation_split=validation_split, epochs=200, batch_size=64,
                            callbacks=callbacks)

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
    plt.show()

    # model.evaluate(x_test, y_test, verbose=1)
