#!/usr/bin/env python3

from keras import Input, Model
from keras.layers import LSTM, Dense, Lambda
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from config.definitions import ROOT_DIR
import numpy as np


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):

    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # return all models
    return model, encoder_model, decoder_model


if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 25
    time_steps_in = 50
    time_steps_out = 50
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    # ADJUSTED NORMALIZATION DATA --------------------------------------------------------------------------------
    # sorted_data_for_learning = SortedDataForLearning(
    #     path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")
    # training_data = sorted_data_for_learning.training_data
    # -------------------------------------------------------------------------------------------------------------

    # UNIVERSAL NORMALIZATION DATA ------------------------------------------------------------------------------
    training_data = np.load(ROOT_DIR + "/data_storage/data/universal_norm/normalized_data.npy")
    # -------------------------------------------------------------------------------------------------------------

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split

    x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0]), time_steps_in, 13))
    y_train = to_categorical(training_data[:, -1])
    x_train = x_train[:, :, 1:]

    y_train_final = np.zeros((len(training_data), time_steps_in, y_train.shape[1]))
    for line in range(0, y_train.shape[0]):
        result = y_train[line, :]
        for t in range(0, time_steps_in):
            y_train_final[line][t] = result

    y_train_middle = np.empty((y_train_final.shape[0], y_train_final.shape[1], y_train_final.shape[2]))
    for _ in range(0, y_train_final.shape[0]):
        for __ in range(0, y_train_final.shape[1]):
            for ___ in range(0, y_train_final.shape[2]):
                if __ == 0:
                    y_train_middle[_][__][___] = start_number
                else:
                    y_train_middle[_][__][___] = y_train_final[_][__][___]

    print("\n")
    print(x_train.shape, y_train_final.shape, y_train_middle.shape)
    print("\n")
    # print('x_train[0]= %s, y_train_final=%s, y_train_middle=%s' % (x_train[0], y_train_final[0], y_train_middle[0]))
    # print("\n")

    train, infenc, infdec = define_models(params, labels, neurons)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_loss', patience=10)
    fit_history = train.fit([x_train, y_train_middle], y_train_final, batch_size=batch_size, epochs=epochs,
                            validation_split=validation_split, shuffle=True, verbose=2, callbacks=[callback])

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

    # train.save("train_model")
    # infenc.save("infenc_model")
    # infdec.save("infdec_model")
