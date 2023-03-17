#!/usr/bin/env python3
import keras.models
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
from progressbar import progressbar
from time import sleep
from keras.layers import Lambda
from keras import backend as K

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
import tensorflow as tf
from neural_networks.utils import plot_confusion_matrix_percentage


def training_encoder_decoder(out_dim, input_params, out_labels):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, input_params), name="encoder_inputs")
    encoder = LSTM(out_dim, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, out_labels), name="decoder_inputs")
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(out_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(out_labels, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Run training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense


def decode_sequence(input_seq, sn, output_dim, o_labels, ei, es, di, dlstm, dd):

    encoder_model = Model(ei, es)

    plot_model(encoder_model, to_file="seq2seq/testing_model_1.png", show_shapes=True)

    decoder_state_input_h = Input(shape=(output_dim,))
    decoder_state_input_c = Input(shape=(output_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = dlstm(di, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = dd(decoder_outputs)
    decoder_model = Model([di] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    plot_model(decoder_model, to_file="seq2seq/testing_model_2.png", show_shapes=True)

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, o_labels))
    # Populate the first character of target sequence with the start character.
    target_seq[:, 0, :] = sn

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_seq = list()
    while not stop_condition:

        # in a loop
        # decode the input to a token/output prediction + required states for context vector
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # convert the token/output prediction to a token/output
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_digit = sampled_token_index

        # add the predicted token/output to output sequence
        decoded_seq.append(sampled_digit)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_seq) == time_steps:
            stop_condition = True

        # Update the input target sequence (of length 1)
        # with the predicted token/output
        target_seq = np.zeros((1, 1, o_labels))
        target_seq[0, 0, sampled_token_index] = sn

        # Update input states (context vector)
        # with the ouputed states
        states_value = [h, c]

        # loop back.....

    # when loop exists return the output sequence
    return decoded_seq


if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 64
    time_steps = 50
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    # sorted_data_for_learning = SortedDataForLearning(
    #     path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/", config_file="training_config_rnn")

    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1 - validation_split)
    n_test = test_data.shape[0]

    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], time_steps, 13))
    y_train = to_categorical(training_data[:, -1])
    x_train = x_train[:, :, 1:]

    results = []
    encodeds = []
    y_train_final = []
    x_train_decoder = []

    for line in range(0, y_train.shape[0]):

        result = y_train[line, :]

        for i in range(0, x_train.shape[1]):
            if i == 0:
                encoded = np.array([start_number, start_number, start_number, start_number], dtype=float)
            else:
                encoded = result
            results.append(result)
            encodeds.append(encoded)

        y_train_final.append(results)
        x_train_decoder.append(encodeds)
        results = []
        encodeds = []

    y_train_final = np.array(y_train_final, dtype=float)
    x_train_decoder = np.array(x_train_decoder, dtype=float)

    training_model, e_inputs, e_states, d_inputs, d_lstm, d_dense = training_encoder_decoder(neurons, params, labels)
    training_model.summary()
    plot_model(training_model, to_file="seq2seq/model.png", show_shapes=True)

    print(x_train.shape)
    print(x_train_decoder.shape)
    print(y_train_final.shape)

    print("x_train_decoder[0, :, :]")
    print(x_train_decoder[0, :, :])
    print("y_train_final[0, :, :]")
    print(y_train_final[0, :, :])

    callback = EarlyStopping(monitor='val_loss', patience=10)
    fit_history = training_model.fit([x_train, x_train_decoder], y_train_final, batch_size=batch_size, epochs=epochs,
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

    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq/training_curves_RNN_adjusted_norm.png", bbox_inches='tight')

    x_test = np.reshape(test_data[:, :-1], (n_test, time_steps, 13))
    y_test = test_data[:, -1]
    x_test = x_test[:, :, 1:]

    y_test_final = []

    for line in range(0, y_test.shape[0]):
        for y in range(0, time_steps):
            y_test_final.append(int(y_test[line]))

    print(x_test.shape)

    predicted = list()
    n_test = 0

    for n in progressbar(range(int(x_test.shape[0]/10)), redirect_stdout=True):
        predicted += decode_sequence(x_test[n][np.newaxis, :, :], start_number, neurons, labels, e_inputs,
                                     e_states, d_inputs, d_lstm, d_dense)
        n_test += time_steps

    cm = confusion_matrix(y_true=y_test_final[0:n_test], y_pred=predicted)
    print("cm")
    print(cm)

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq/confusion_matrix.png", bbox_inches='tight')
    cm_true = cm / cm.astype(float).sum(axis=1)
    cm_true_percentage = cm_true * 100
    plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
                                     title="Confusion Matrix (%) - Seq-To-Seq")
    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq/confusion_matrix_true.png", bbox_inches='tight')


    # metric_accuracy = (tp + tn) / (fp + fn + tp + tn)
    # metric_recall = tp / (fn + tp)
    # metric_precision = tp / (fp + tp)
    # metric_f1 = 2 * (metric_precision * metric_recall) / (metric_precision + metric_recall)