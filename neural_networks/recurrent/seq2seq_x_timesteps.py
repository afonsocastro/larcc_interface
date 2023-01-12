#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras import Input, Model
import tensorflow as tf
from keras.callbacks import EarlyStopping
# from keras.utils import to_categorical  # one-hot encode target column
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from keras.layers import Dense, LSTM, Layer
from keras.utils.vis_utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from progressbar import progressbar
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
from neural_networks.utils import plot_confusion_matrix_percentage
from keras.layers import Lambda
from keras import backend as K


class BahdanauAttention(Layer):
    def __init__(self, units, verbose=0):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.verbose = verbose

    def call(self, query, values):
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        if self.verbose:
            print('score: (batch_size, max_length, 1) ', score.shape)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ', attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ', context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ', context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
        return context_vector, attention_weights


def training_encoder_decoder(out_dim, input_params, out_labels, start_n, batch_s, time_ss):
    # The first part is encoder
    encoder_inputs = Input(shape=(None, input_params), name='encoder_inputs')
    encoder_lstm = LSTM(out_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)

    # initial context vector is the states of the encoder
    encoder_states = [encoder_state_h, encoder_state_c]

    # Set up the attention layer
    attention = BahdanauAttention(out_dim)

    # Set up the decoder layers
    decoder_inputs = Input(shape=(1, (out_labels + out_dim)), name='decoder_inputs')
    decoder_lstm = LSTM(out_dim, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(out_labels, activation='softmax', name='decoder_dense')

    all_outputs = []

    # 1 initial decoder's input data
    # Prepare initial decoder input data that just contains the start character
    # Note that we made it a constant one-hot-encoded in the model
    # that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
    # one-hot encoded zero(0) is the start symbol

    inputs = np.zeros((batch_s, 1, out_labels), dtype="float32")
    inputs[:, :, :] = start_n

    # 2 initial decoder's state
    # encoder's last hidden state + last cell state
    decoder_outputs = encoder_state_h
    states = encoder_states

    # decoder will only process one time step at a time.
    for _ in range(time_ss):
        # 3 pay attention
        # create the context vector by applying attention to
        # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
        context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)

        context_vector = tf.expand_dims(context_vector, 1)

        # 4. concatenate the input + context vectore to find the next decoder's input
        inputs = tf.concat([context_vector, inputs], axis=-1)

        # 5. passing the concatenated vector to the LSTM
        # Run the decoder on one timestep with attended input and previous states
        decoder_outputs, state_h, state_c = decoder_lstm(inputs,
                                                         initial_state=states)
        # decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

        outputs = decoder_dense(decoder_outputs)
        # 6. Use the last hidden state for prediction the output
        # save the current prediction
        # we will concatenate all predictions later
        outputs = tf.expand_dims(outputs, 1)
        all_outputs.append(outputs)
        # 7. Reinject the output (prediction) as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]

    # 8. After running Decoder for max time steps
    # we had created a predition list for the output sequence
    # convert the list to output array by Concatenating all predictions
    # such as [batch_size, timesteps, features]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # 9. Define and compile model
    model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 64
    # time_steps = 50
    time_steps = 100
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 100
    # epochs = 50

    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * validation_split
    n_val = len(training_data) * (1 - validation_split)
    n_test = test_data.shape[0]

    x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0] / 2), time_steps, 13))

    y_train = to_categorical(training_data[:, -1])
    x_train = x_train[:, :, 1:]

    results = []
    # encodeds = []
    y_train_final = []
    # x_train_decoder = []
    for line in range(0, y_train.shape[0], 2):

        result1 = y_train[line, :]
        result2 = y_train[line+1, :]

        for i in range(0, int(x_train.shape[1] / 2)):
            # if i == 0:
            #     encoded1 = np.array([start_number, start_number, start_number, start_number], dtype=float)
            # else:
            #     encoded1 = result1

            results.append(result1)
            # encodeds.append(encoded1)

        for i in range(int(x_train.shape[1] / 2), x_train.shape[1]):
            # if i == int(x_train.shape[1] / 2):
            #     encoded2 = encoded1
            # else:
            #     encoded2 = result2
            results.append(result2)
            # encodeds.append(encoded2)

        y_train_final.append(results)
        # x_train_decoder.append(encodeds)
        results = []
        # encodeds = []

    y_train_final = np.array(y_train_final, dtype=float)
    # x_train_decoder = np.array(x_train_decoder, dtype=float)
    # x_train_decoder[:, 0, :] = start_number

    # training_model, e_inputs, e_states, d_inputs, d_lstm, d_dense = training_encoder_decoder(neurons, params, labels)
    model_encoder_decoder_Bahdanau_Attention = training_encoder_decoder(neurons, params, labels, start_number, batch_size, time_steps)
    model_encoder_decoder_Bahdanau_Attention.summary()

    # plot_model(model_encoder_decoder_Bahdanau_Attention, to_file="seq2seq/model.png", show_shapes=True)

    x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
    y_test = test_data[:, -1]
    x_test = x_test[:, :, 1:]

    # print("x_test.shape")
    # print(x_test.shape)
    # print("x_test[0].shape")
    # print(x_test[0].shape)
    # print("x_test[0]")
    # print(x_test[0])
    #
    # pred = model_encoder_decoder_Bahdanau_Attention.predict(x_test[0].reshape(1, x_test[0].shape[0], x_test[0].shape[1]))
    # print('input: ', x_test[0])
    # print('expected: ',y_test[0])
    # print('predicted: ', pred[0])

    print("x_train.shape")
    print(x_train.shape)
    print("y_train_final.shape")
    print(y_train_final.shape)

    callback = EarlyStopping(monitor='val_loss', patience=10)
    fit_history = model_encoder_decoder_Bahdanau_Attention.fit(x_train, y_train_final, batch_size=batch_size,
                                                               epochs=epochs, validation_split=validation_split,
                                                               shuffle=True, verbose=2, callbacks=[callback])
    exit(0)
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

    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq/training_curves.png", bbox_inches='tight')

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

    predicted = list()
    n_test = 0

    for n in progressbar(range(int(x_test.shape[0]/10)), redirect_stdout=True):
        new_predicted = decode_sequence(x_test[n][np.newaxis, :, :], start_number, neurons, labels, e_inputs,
                                     e_states, d_inputs, d_lstm, d_dense)

        predicted += new_predicted
        print("new_predicted")
        print(new_predicted)

        print("len(new_predicted)")
        print(len(new_predicted))


        n_test += time_steps

        print("y_test_final[0:n_test]")
        print(y_test_final[n_test - time_steps:n_test])

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
