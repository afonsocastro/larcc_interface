#!/usr/bin/env python3

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras
import keras_nlp
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

from config.definitions import ROOT_DIR


def build_model(input_shape, head_size, num_heads, n_classes):
    inputs = keras.Input(shape=input_shape)
    positional_encoding = keras_nlp.layers.SinePositionEncoding()(inputs)
    outputs = inputs + positional_encoding
    # outputs = inputs
    # x = outputs

    x = layers.LayerNormalization(epsilon=1e-6)(outputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=0)(x, x)
    res = x + outputs

    # x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = Reshape((20, 12, 1), input_shape=(20, 12))(res)
    x = Conv2D(64, kernel_size=(5, 1), activation="relu", input_shape=(20, 12, 1))(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(32, kernel_size=(2, 1), activation="relu")(x)
    x = MaxPooling2D((2, 1))(x)

    # x = layers.Dense(12, activation="relu", kernel_regularizer='l1')(x)

    # transformer_output = x + res
    transformer_output = x

    flatten_output = layers.Flatten()(transformer_output)
    dense_output = layers.Dense(units=n_classes, activation="softmax", kernel_regularizer='l1')(flatten_output)
    model = keras.models.Model(inputs=inputs, outputs=dense_output)

    return model


if __name__ == '__main__':
    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    validation_split = 0.3

    training_data = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_train_data_20ms.npy")

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], time_steps, 13))
    y_train = to_categorical(training_data[:, -1])

    x_train = x_train[:, :, 1:]
    print(x_train.shape)
    print(y_train.shape)

    input_shape = x_train.shape[1:]

    print("input_shape")
    print(input_shape)

    model = build_model(input_shape, head_size=16, num_heads=4, n_classes=n_labels)
    # , )

    # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #               metrics=["sparse_categorical_accuracy"])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file="tst_model.png", show_shapes=True)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)]

    fit_history = model.fit(x_train, y_train, validation_split=validation_split, epochs=500, batch_size=64,
                            callbacks=callbacks)
    model.save("transformer_model")
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
