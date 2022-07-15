#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout

if __name__ == '__main__':
    validation_split = 0.3

    all_data = np.load('../data/learning_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
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
        Dense(units=16, input_shape=(650,), activation='relu'),
        Dense(units=32, activation='relu'),
        Dropout(0.5),
        Dense(units=4, activation='softmax')
    ])

    # `rankdir='LR'` is to make the graph horizontal.
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=all_data[:, :-1], y=all_data[:, -1], validation_split=validation_split, batch_size=5, shuffle=True, epochs=50,
              verbose=2)

    print("\n")
    print("Using %d samples for training and %d for validation" % (len(all_data) - validation_n, validation_n))
    print("\n")

    model.save("myModel")

    # predictions = model.predict(x=all_data[27:28, :-1], verbose=2)
    #
    # print("\n")
    # print("predictions")
    # print(predictions)
    #
    # print("\n")
    #
    # print("all_data[27:28, -1]")
    # print(all_data[27:28, -1])
