#!/usr/bin/env python3
import keras.models
from keras import Input, Model
import tensorflow as tf
from keras.layers import Dense, LSTM, Layer
from config.definitions import ROOT_DIR
from data_storage.full_timewindow.src.ProcessData import ProcessData
import numpy as np
from neural_networks.utils import plot_confusion_matrix_percentage
from keras.layers import Lambda
from keras import backend as K
from numpy import save, delete


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
    batch_size = 2
    # time_steps = 50
    time_steps = 6000
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50
    # epochs = 50

    # delete('pred_model_Bahdanau_Attention.npy')
    # delete('true_model_Bahdanau_Attention.npy')

    processed_data = ProcessData(path=ROOT_DIR + "/data_storage/full_timewindow/data/",
                                 data_file="raw_learning_data.npy", div=0.7)

    test_data = processed_data.test_data_60s

    n_test = test_data.shape[0]

    model_Bahdanau_Attention2 = keras.models.load_model("model_Bahdanau_Attention2")

    x_test = test_data[:, :, 1:-1]
    y_test = test_data[:, :,  -1]

    print("x_test.shape")
    print(x_test.shape)
    print("y_test.shape")
    print(y_test.shape)


    # y_test_final = []
    # for sample in range(0, y_test.shape[0]):
    #     for ts in range(0, y_test.shape[1]):
    #         if y_test[sample][ts]
    #         r = [int(y_test[line]), int(y_test[line + 1])]
    #     y_test_final.append(r)

    # nv = x_test[0:18].reshape(18, time_steps, params)
    # print(nv.shape)
    # exit(0)

    pred = model_Bahdanau_Attention2.predict(x_test, batch_size=2)

    print("type(pred)")
    print(type(pred))
    print("pred.shape")
    print(pred.shape)
    # exit(0)

    save('pred_model_Bahdanau_Attention2.npy', pred)
    save('true_model_Bahdanau_Attention2.npy', y_test[0:18])

    # print('input: ', x_test[0:2].shape)
    # print('expected: ', y_test_final)
    # print('predicted: ', pred[0:2])
