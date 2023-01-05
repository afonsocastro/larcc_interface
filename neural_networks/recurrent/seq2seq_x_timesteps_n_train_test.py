#!/usr/bin/env python3

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical  # one-hot encode target column
from keras.layers import Dense, LSTM
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
from neural_networks.utils import NumpyArrayEncoder, prediction_classification
import numpy as np
import json
from progressbar import progressbar


def training_encoder_decoder(out_dim, input_params, out_labels):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, input_params), name="encoder_inputs")
    encoder = LSTM(out_dim, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, out_labels), name="decoder_inputs")
    decoder_lstm = LSTM(out_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(out_labels, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

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
    seq = list()
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
        seq.append(output_tokens[0, -1, :].tolist())

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
    return decoded_seq, seq


if __name__ == '__main__':
    n_times = 20

    validation_split = 0.3
    batch_size = 64
    # time_steps = 50
    time_steps = 100
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 100

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    training_test_list = []
    for n in progressbar(range(n_times), redirect_stdout=True):

        sorted_data_for_learning = SortedDataForLearning(
            path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

        training_data = sorted_data_for_learning.training_data
        test_data = sorted_data_for_learning.test_data

        n_train = len(training_data) * (1 - validation_split)
        n_val = len(training_data) * validation_split
        n_test = test_data.shape[0]

        x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0] / 2), time_steps, 13))

        y_train = to_categorical(training_data[:, -1])
        x_train = x_train[:, :, 1:]

        results = []
        y_train_final = []
        x_train_decoder = []
        for line in range(0, y_train.shape[0], 2):

            result1 = y_train[line, :]
            result2 = y_train[line+1, :]

            for i in range(0, int(x_train.shape[1] / 2)):
                results.append(result1)
            for i in range(int(x_train.shape[1] / 2), x_train.shape[1]):
                results.append(result2)

            y_train_final.append(results)
            x_train_decoder.append(results)
            results = []

        y_train_final = np.array(y_train_final, dtype=float)
        x_train_decoder = np.array(x_train_decoder, dtype=float)
        x_train_decoder[:, 0, :] = start_number

        training_model, e_inputs, e_states, d_inputs, d_lstm, d_dense = training_encoder_decoder(neurons, params, labels)
        training_model.summary()
        plot_model(training_model, to_file="seq2seq/model.png", show_shapes=True)

        print(x_train.shape)
        print(x_train_decoder.shape)
        print(y_train_final.shape)
        callback = EarlyStopping(monitor='val_loss', patience=10)
        fit_history = training_model.fit([x_train, x_train_decoder], y_train_final, batch_size=batch_size, epochs=epochs,
                                         validation_split=validation_split, shuffle=True, verbose=2, callbacks=[callback])

        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print("TRAINING %d time" % n)
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        print("\n")
        print("Using %d samples for training and %d for validation" % (n_train, n_val))
        print("\n")

        print("\n")
        print("-------------------------------------------------------------------------------------------------")
        print("TESTING %d time" % n)
        print("-------------------------------------------------------------------------------------------------")
        print("\n")

        predictions_list = []

        pull = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
        push = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
        shake = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                 "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
        twist = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
                 "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}

        x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
        y_test = test_data[:, -1]
        x_test = x_test[:, :, 1:]

        y_test_final = []

        for line in range(0, y_test.shape[0], 2):

            for y in range(0, int(time_steps / 2)):
                y_test_final.append(int(y_test[line]))
            for y in range(int(time_steps / 2), time_steps):
                y_test_final.append(int(y_test[line+1]))

        print(x_test.shape)

        predicted_decoded = list()
        predicted_seq = list()
        n_test = 0

        for n in progressbar(range(x_test.shape[0]), redirect_stdout=True):
            decoded_seq, seq = decode_sequence(x_test[n][np.newaxis, :, :], start_number, neurons, labels, e_inputs,
                                         e_states, d_inputs, d_lstm, d_dense)

            predicted_decoded += decoded_seq
            for ts in range(0, time_steps):
                predicted_seq.append(np.reshape(np.array(seq[ts]), (1, 4)))

            n_test += time_steps

        for i in range(0, len(predicted_decoded)):
            true = y_test_final[i]
            decoded_prediction = predicted_decoded[i]
            sequence_prediction = predicted_seq[i]

            prediction_classification(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull,
                                      pred=sequence_prediction)
            prediction_classification(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push,
                                      pred=sequence_prediction)
            prediction_classification(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake,
                                      pred=sequence_prediction)
            prediction_classification(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist,
                                      pred=sequence_prediction)

            predictions_list.append(decoded_prediction)

        cm = confusion_matrix(y_true=y_test_final[0:n_test], y_pred=predicted_decoded)
        cm_true = cm / cm.astype(float).sum(axis=1)

        test_dict = {"cm_true": cm_true, "cm": cm, "pull": pull, "push": push, "shake": shake, "twist": twist}
        training_test_dict = {"training": fit_history.history, "test": test_dict}
        training_test_list.append(training_test_dict)

        with open(
                ROOT_DIR + "/neural_networks/recurrent/seq2seq_x_timesteps_n_times/training_testing_n_times_seq2seq_x_timesteps.json",
                "w") as wf:
            json.dump(training_test_list, wf, cls=NumpyArrayEncoder)
