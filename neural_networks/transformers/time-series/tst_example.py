#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import TimeDistributed, Conv1D, GlobalMaxPooling1D


# Define the TST model as a custom Keras layer
class TST(tf.keras.layers.Layer):
    def __init__(self, n_heads=8, d_model=32, d_ff=128, seq_len=None, dropout_rate=0.1, **kwargs):
        super(TST, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate

        self.attention = MultiHeadAttention(n_heads, d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.conv1 = Conv1D(filters=d_ff, kernel_size=1, activation='relu')
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.dense = TimeDistributed(Dense(1))
        self.pooling = GlobalMaxPooling1D()

    def call(self, inputs):
        # Add positional encoding to input
        pos_enc = tf.expand_dims(tf.range(self.seq_len, dtype=tf.float32), axis=0)
        pos_enc = tf.tile(pos_enc, [inputs.shape[0], 1])
        inputs = tf.concat([inputs, pos_enc], axis=-1)

        # Apply self-attention and residual connections
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output)
        attention_output = self.norm1(attention_output + inputs)

        # Apply convolutional layer and residual connections
        conv_output = self.conv1(attention_output)
        conv_output = self.dropout2(conv_output)
        conv_output = self.norm2(conv_output + attention_output)

        # Apply dense and pooling layers
        dense_output = self.dense(conv_output)
        output = self.pooling(dense_output)

        return output


# Define the TST model architecture
def build_model(seq_len):
    inputs = Input(shape=(seq_len, 1))
    x = TST(seq_len=seq_len)(inputs)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    # Generate some sample data
    seq_len = 50
    n_samples = 1000
    x = np.random.randn(n_samples, seq_len, 1)
    y = np.random.randn(n_samples, 1)

    print("x")
    print(x.shape)
    # print(x)
    print("y")
    print(y.shape)
    # print(y)

    # Build and compile the model
    model = build_model(seq_len)
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    model.fit(x, y, epochs=10, batch_size=32)

