#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras import Input, Model
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout, concatenate  # create model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage

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

    # model = Sequential()
    # model.add(LSTM(16, input_shape=(time_steps, params), return_sequences=True))
    # # model.add(LSTM(16, input_shape=(time_steps, params)))
    # model.add(LSTM(16, return_state=True))
    # model.add(Dense(4, activation="softmax"))

    input = Input(shape=(time_steps, params))
    lstm1 = LSTM(16, return_sequences=True)(input)
    lstm2, state_h2, state_c2 = LSTM(16, return_state=True)(lstm1)
    combined_input = concatenate([lstm2, state_c2])
    output = Dense(4, activation='softmax')(combined_input)

    model = Model(inputs=input, outputs=[output])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file="seq2label_model.png", show_shapes=True)

    callback = EarlyStopping(monitor='val_loss', patience=20)

    fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, batch_size=batch_size,
                            epochs=epochs, shuffle=True, callbacks=[callback], verbose=2)
    model.save("seq2label_20ms")
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
