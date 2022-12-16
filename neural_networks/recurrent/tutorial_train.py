#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical  # one-hot encode target column
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, MaxPooling2D, Dropout, Lambda  # create model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.layers import Lambda
from keras import backend as K

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage


def create_hard_coded_decoder_input_model(batch_size):
    encoder_inputs = Input(shape=(50, 2), name="encoder_inputs")
    encoder_lstm = LSTM(16, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    states = [state_h, state_c]

    decoder_inputs = Input(shape=(1, 2), name="decoder_inputs")
    decoder_lstm = LSTM(16, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_dense = Dense(2, activation="softmax", name="decoder_dense")

    all_outputs = []

    decoder_input_data = np.ones((batch_size, 1, 2))
    decoder_input_data[:, :, :] = 50
    inputs = decoder_input_data

    for _ in range(50):

        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        outputs = decoder_dense(outputs)
        all_outputs.append(outputs)

        inputs = outputs
        states = [state_h, state_c]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    model = Model(encoder_inputs, decoder_outputs, name="model_encoder_decoder")
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# @title Function to Train & Test  given model (Early Stopping monitor 'val_loss')
def train_test(model, X_train, y_train, X_test, y_test, epochs=500, batch_size=32, patience=5, verbose=0):
    # patient early stopping
    # es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, patience=20)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    # train model
    print('training for ', epochs, ' epochs begins with EarlyStopping(monitor= val_loss, patience=', patience, ')....')
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        callbacks=[es])
    print(epochs, ' epoch training finished...')

    # report training
    # list all data in history
    # print(history.history.keys())
    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print('\nPREDICTION ACCURACY (%):')
    print('Train: %.3f, Test: %.3f' % (train_acc * 100, test_acc * 100))
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(model.name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model.name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # spot check some examples
    space = 3 * len(y_test[0])
    print('10 examples from test data...')
    print('Input', ' ' * (space - 4), 'Expected', ' ' * (space - 7),
          'Predicted', ' ' * (space - 5), 'T/F')
    correct = 0
    sampleNo = 10

    predicted = model_encoder_decoder.predict(X_test[:sampleNo], batch_size=batch_size)
    for sample in range(0, sampleNo):
        if y_test[sample] == predicted[sample]:
            correct += 1
        print(X_test[sample], ' ', y_test[sample], ' ', predicted[sample], ' ', y_test[sample] == predicted[sample])
        print('Accuracy: ', correct / sampleNo)


if __name__ == '__main__':
    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/", config_file="training_config_rnn")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    validation_split = 0.3
    batch_size = 64
    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1 - validation_split)
    n_test = test_data.shape[0]

    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], 50, 2))
    y_train = to_categorical(training_data[:, -1])

    y_train_final = []
    results = []
    for line in range(0, y_train.shape[0]):

        result = y_train[line, :]
        for i in range(0, x_train.shape[1]):
            results.append(result)

        y_train_final.append(results)
        results = []

    y_train_final = np.array(y_train_final, dtype=float)

    x_test = np.reshape(test_data[:, :-1], (n_test, 50, 2))
    y_test = to_categorical(test_data[:, -1])

    y_test_final = []
    results = []
    for line in range(0, y_test.shape[0]):

        result = y_test[line, :]
        for i in range(0, x_test.shape[1]):
            results.append(result)

        y_test_final.append(results)
        results = []

    y_test_final = np.array(y_test_final, dtype=float)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train_final.shape)
    print(y_test_final.shape)

    # context_vector = model_encoder(x_train[0][np.newaxis, :, :])

    model_encoder_decoder = create_hard_coded_decoder_input_model(batch_size=batch_size)
    model_encoder_decoder.summary()
    plot_model(model_encoder_decoder, show_shapes=True)

    # print(x_train[0].shape)
    #
    # for _ in range(len(x_train)):
    #     print("here")
    #     print(_)
    #     x_train[_] = x_train[_][np.newaxis, :, :]
    #     y_train_final[_] = y_train_final[_][np.newaxis, :, :]
    # print(x_train[0].shape)

    # print(len(x_train))
    # print(x_train.shape)
    # print(y_train_final.shape)
    # print(model_encoder_decoder.output_shape)
    # print(model_encoder_decoder.input_shape)

    # print(x_train[0].shape)
    # x_train[0] = x_train[0][np.newaxis, :, :]
    # print(x_train[0][np.newaxis, :, :])
    output = model_encoder_decoder(x_train[0][np.newaxis, :, :])
    print(output.shape)
    print(output)
    print(y_train_final[0])
    print(y_train_final[0].shape)

    model_encoder_decoder.fit(x_train[np.newaxis, :, :], y_train_final[np.newaxis, :, :], batch_size=batch_size, epochs=30, validation_split=validation_split)
    #
    # train_test(model_encoder_decoder, x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=500, verbose=1)

    # model = Sequential()
    # model.add(LSTM(64, batch_input_shape=(None, None, 2), return_sequences=False))
    # # Dropout(0.2)
    # # model.add(LSTM(1))
    # model.add(Dense(2, activation="softmax"))
    #
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # # model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
    #
    # model.summary()
    # # exit(0)
    #
    # # fit_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=400)
    # fit_history = model.fit(x=x_train, y=y_train, validation_split=validation_split, epochs=25, verbose=2)
    #
    # fig = plt.figure()
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(fit_history.history['accuracy'])
    # plt.plot(fit_history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(fit_history.history['loss'])
    # plt.plot(fit_history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    #
    # plt.show()

    # results = model.predict(x_test)

    # results = np.argmax(results, axis=1, out=None)
    # y_results = np.argmax(y_test, axis=1, out=None)

    # print(results.shape)
    # print(y_test.shape)
    #
    # print(results)
    # print(y_test)

    # print()
    # plt.scatter(range(results.shape[0]), results, color='r')
    # plt.scatter(range(results.shape[0]), y_test, color='g')
    # plt.show()

    # data = [[[(i + j) / 100] for i in range(6)] for j in range(100)]
    # target = [(i + 6) / 100 for i in range(100)]
    #
    # data = np.array(data, dtype=float)
    # target = np.array(target, dtype=float)
    #
    # x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
    # fit_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=400)
    # # model.save("myModel")
    #
    # results = model.predict(x_test)
    #
    # # results = np.argmax(predicted_values, axis=1, out=None)
    # # y_results = np.argmax(y_test, axis=1, out=None)
    #
    # plt.scatter(range(results.shape[0]), results, color='r')
    # plt.scatter(range(results.shape[0]), y_test, color='g')
    # plt.show()





