#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import random
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model


import collections
import math
import string

import numpy as np
import tensorflow.compat.v2 as tf

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.layers import activation
from keras.layers import core
from keras.layers import regularization
from keras.utils import tf_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning

from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.backend import softmax

_CHR_IDX = string.ascii_lowercase


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = f"{input_str},{kernel_str}->{output_str}"

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.

    Once the method is called, self._built_from_signature will be set to
    True.

    Args:
        query: Query tensor or TensorShape.
        value: Value tensor or TensorShape.
        key: Key tensor or TensorShape.
    """
    self._built_from_signature = True
    if hasattr(query, "shape"):
        self._query_shape = tf.TensorShape(query.shape)
    else:
        self._query_shape = tf.TensorShape(query)
    if hasattr(value, "shape"):
        self._value_shape = tf.TensorShape(value.shape)
    else:
        self._value_shape = tf.TensorShape(value)
    if key is None:
        self._key_shape = self._value_shape
    elif hasattr(key, "shape"):
        self._key_shape = tf.TensorShape(key.shape)
    else:
        self._key_shape = tf.TensorShape(key)

    # Any setup work performed only once should happen in an `init_scope`
    # to avoid creating symbolic Tensors that will later pollute any eager
    # operations.
    with tf_utils.maybe_init_scope(self):
        free_dims = self._query_shape.rank - 1
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=1, output_dims=2
        )
        self._query_dense = core.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            self._key_shape.rank - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = core.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            self._value_shape.rank - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = core.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports mult-head einsum computations.
        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(
            free_dims,
            self._get_common_kwargs_for_sublayer(),
            "attention_output",
        )


def scaled_dot_product_attention(Q, K, V, num_heads, d_model, mask=None):
    """
    Args:
        Q (tf.tensor): of shape (h * batch, q_size, d_model)
        K (tf.tensor): of shape (h * batch, k_size, d_model)
        V (tf.tensor): of shape (h * batch, k_size, d_model)
        mask (tf.tensor): of shape (h * batch, q_size, k_size)
    """

    d = d_model // num_heads
    assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

    out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
    out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

    if mask is not None:
        # masking out (0.0) => setting to -inf.
        out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

    out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
    # out = tf.layers.dropout(out, training=self._is_training)
    out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

    return out


def multihead_attention(num_heads, d_model, query, memory=None, mask=None, scope='attn'):
    """
    Args:
        query (tf.tensor): of shape (batch, q_size, d_model)
        memory (tf.tensor): of shape (batch, m_size, d_model)
        mask (tf.tensor): shape (batch, q_size, k_size)

    Returns:h
        a tensor of shape (bs, q_size, d_model)
    """
    _query_dense = core.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(
            output_rank - 1, [self._num_heads, self._key_dim]
        ),
        bias_axes=bias_axes if self._use_bias else None,
        name="query",
        **self._get_common_kwargs_for_sublayer(),
    )

    if memory is None:
        memory = query

    # # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
    # Q = Dense(query, d_model, activation="relu")
    # K = Dense(memory, d_model, activation=tf.nn.relu)
    # V = Dense(memory, d_model, activation=tf.nn.relu)
    #
    # # Split the matrix to multiple heads and then concatenate to have a larger
    # # batch size: [h*batch, q_size/k_size, d_model/num_heads]
    # Q_split = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    # K_split = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    # V_split = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
    # mask_split = tf.tile(mask, [num_heads, 1, 1])

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(memory)

    # `value` = [B, S, N, H]
    value = self._value_dense(memory)

    # Apply scaled dot product attention
    out = scaled_dot_product_attention(Q_split, K_split, V_split, num_heads=num_heads, d_model=d_model,
                                       mask=mask_split)

    # Merge the multi-head back to the original shape
    out = tf.concat(tf.split(out, num_heads, axis=0), axis=2)  # [bs, q_size, d_model]

    # The final linear layer and dropout.
    # out = tf.layers.dense(out, self.d_model)
    # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)

    return out


# # Implementing the Scaled-Dot Product Attention
# class DotProductAttention(Layer):
#     def __init__(self, **kwargs):
#         super(DotProductAttention, self).__init__(**kwargs)
#
#     def call(self, queries, keys, values, d_k, mask=None):
#         # Scoring the queries against the keys after transposing the latter, and scaling
#         scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
#
#         # Apply mask to the attention scores
#         if mask is not None:
#             scores += -1e9 * mask
#
#         # Computing the weights by a softmax operation
#         weights = softmax(scores)
#
#         # Computing the attention by a weighted sum of the value vectors
#         return matmul(weights, values)


# # Implementing the Multi-Head Attention
# class MultiHeadAttention(Layer):
#     def __init__(self, h, d_k, d_v, d_model, **kwargs):
#         super(MultiHeadAttention, self).__init__(**kwargs)
#         self.attention = DotProductAttention()  # Scaled dot product attention
#         self.heads = h  # Number of attention heads to use
#         self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
#         self.d_v = d_v  # Dimensionality of the linearly projected values
#         self.d_model = d_model  # Dimensionality of the model
#         self.W_q = Dense(d_k)  # Learned projection matrix for the queries
#         self.W_k = Dense(d_k)  # Learned projection matrix for the keys
#         self.W_v = Dense(d_v)  # Learned projection matrix for the values
#         self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output
#
#     def reshape_tensor(self, x, heads, flag):
#         if flag:
#             # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
#             x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
#             x = transpose(x, perm=(0, 2, 1, 3))
#         else:
#             # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
#             x = transpose(x, perm=(0, 2, 1, 3))
#             x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
#         return x
#
#     def call(self, queries, keys, values, mask=None):
#         # Rearrange the queries to be able to compute all heads in parallel
#         q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
#         # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
#
#         # Rearrange the keys to be able to compute all heads in parallel
#         k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
#         # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
#
#         # Rearrange the values to be able to compute all heads in parallel
#         v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
#         # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
#
#         # Compute the multi-head attention output using the reshaped queries, keys and values
#         o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
#         # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
#
#         # Rearrange back the output into concatenated form
#         output = self.reshape_tensor(o_reshaped, self.heads, False)
#         # Resulting tensor shape: (batch_size, input_seq_length, d_v)
#
#         # Apply one final linear projection to the output to generate the multi-head attention
#         # Resulting tensor shape: (batch_size, input_seq_length, d_model)
#         return self.W_o(output)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    d_model = head_size * num_heads  # Dimensionality of the model
    x = multihead_attention(num_heads=num_heads, d_model=d_model, query=x, memory=x)

    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, n_classes, dropout=0,
                mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150

    num_heads = 4
    head_size = 16


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

    input_shape = x_train.shape[1:]

    model = build_model(input_shape, head_size=head_size, num_heads=num_heads, ff_dim=1, num_transformer_blocks=1,
                        n_classes=n_labels, mlp_units=[2])

    # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #               metrics=["sparse_categorical_accuracy"])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file="tst_model.png", show_shapes=True)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)]

    fit_history = model.fit(x_train, y_train, validation_split=validation_split, epochs=500, batch_size=64,
                            callbacks=callbacks)
    model.save("transformer_model")
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

    # model.evaluate(x_test, y_test, verbose=1)
