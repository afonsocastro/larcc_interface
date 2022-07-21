#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
import matplotlib.pyplot as plt

if __name__ == '__main__':
    validation_split = 0.3

    all_data = np.load('../data/learning_data_training.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    print("type(all_data)")
    print(type(all_data))

    # val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    # train_dataframe = dataframe.drop(val_dataframe.index)

    validation_n = len(all_data) * validation_split


    all_data = np.array(all_data)
    # all_data = all_data.reshape(-1, 1)
    #

    model = Sequential([
        Dense(units=64, input_shape=(650,), activation='relu'),
        Dropout(0.1),
        Dense(units=64, activation='relu'),
        Dropout(0.1),
        # Dense(units=64, activation='relu'),
        # Dropout(0.1),
        Dense(units=4, activation='softmax')
    ])

    # `rankdir='LR'` is to make the graph horizontal.
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    fit_history = model.fit(x=all_data[:, :-1], y=all_data[:, -1], validation_split=validation_split, batch_size=64,
                            # shuffle=True, epochs=300, verbose=2)
                            shuffle=True, epochs=300, verbose=2, callbacks=[callback])

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
    print("Using %d samples for training and %d for validation" % (len(all_data) - validation_n, validation_n))
    print("\n")

    model.save("myModel")

