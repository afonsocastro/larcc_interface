#!/usr/bin/env python3
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning

if __name__ == '__main__':
    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150
    validation_split = 0.7
    time_window = 2
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    config_file = "training_config_time_" + str(time_window)
    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1 - validation_split)
    n_test = test_data.shape[0]

    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], time_steps, 13))
    y_train = to_categorical(training_data[:, -1])
    x_test = np.reshape(test_data[:, :-1], (n_test, time_steps, 13))
    y_test = to_categorical(test_data[:, -1])

    x_train = x_train[:, :, 1:]
    x_test = x_test[:, :, 1:]

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    # Define the model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=(time_steps, params)))
    model.add(LSTM(16))
    model.add(Dense(n_labels, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.summary()
    # plot_model(model, to_file="seq2label/model.png", show_shapes=True)

    callback = EarlyStopping(monitor='val_loss', patience=10)

    fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, batch_size=batch_size,
                            epochs=epochs, shuffle=True, callbacks=[callback], verbose=2)
    model.save("bidirectional_20ms")
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

