#!/usr/bin/env python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based multi-head attention layer."""


import collections
import math
import string

import numpy as np
import tensorflow.compat.v2 as tf

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.activations import softmax
from keras.engine.base_layer import Layer
from keras.layers import activation, Dense, Attention
from keras.layers import core
from keras.layers import regularization
from keras.utils import tf_utils
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

_CHR_IDX = string.ascii_lowercase


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, W_q, W_k, W_v, _num_heads, key_dim, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

        self.W_q = W_q
        self.W_k = W_k
        self.W_v = W_v
        self._num_heads = _num_heads
        self._key_dim = key_dim
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)

    def __call__(self, query, key, value, d_k):
        print("queries.shape")
        print(query.shape)
        print("keys.shape")
        print(key.shape)
        # # Rearrange the queries to be able to compute all heads in parallel
        # q_reshaped = self.reshape_tensor(self.W_q(query), self._num_heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        # k_reshaped = self.reshape_tensor(self.W_k(key), self._num_heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        # v_reshaped = self.reshape_tensor(self.W_v(value), self._num_heads, True)

        q_reshaped = self.reshape_tensor(self.W_q(query), self._num_heads, False)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(key), self._num_heads, False)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(value), self._num_heads, False)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
        print("q_reshaped.shape")
        print(q_reshaped.shape)
        print("k_reshaped.shape")
        print(k_reshaped.shape)
        print("v_reshaped.shape")
        print(v_reshaped.shape)
        # print("d_k.shape")
        # print(d_k.shape)
        # Scoring the queries against the keys after transposing the latter, and scaling
        # scores = matmul(q_reshaped, k_reshaped, transpose_b=True) / math.sqrt(cast(d_k, float32))
        # print("scores.shape")
        # print(scores.shape)

        print("tf.expand_dims(q_reshaped, axis=2).shape")
        print(tf.expand_dims(q_reshaped, axis=3).shape)
        print("tf.expand_dims(k_reshaped, axis=1).shape")
        print(tf.expand_dims(k_reshaped, axis=2).shape)

        features = tf.expand_dims(q_reshaped, axis=3) + tf.expand_dims(k_reshaped, axis=2)
        print("features.shape")
        print(features.shape)
        features = tf.nn.tanh(features)
        print("features.shape")
        print(features.shape)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        print("scores.shape")
        print(scores.shape)
        return scores

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, self._key_dim))
            x = transpose(x, perm=(0, 2, 1, 3))

            # x = transpose(x, perm=(0, 2, 1, 3))
            # x = reshape(x, shape=(shape(x)[0], shape(x)[1], self._key_dim))
        return x


class AdditiveAttention(tf.keras.layers.Layer):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.num_hiddens = num_hiddens
        # self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, queries, keys):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting

        print("queries.shape")
        print(queries.shape)
        print("keys.shape")
        print(keys.shape)

        print("tf.expand_dims(q_reshaped, axis=2).shape")
        print(tf.expand_dims(queries, axis=2).shape)
        print("tf.expand_dims(k_reshaped, axis=1).shape")
        print(tf.expand_dims(keys, axis=1).shape)


        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(keys, axis=1)
        print("features.shape")
        print(features.shape)

        features = self.reshape_tensor(features, self.num_hiddens, False)
        print("features.shape")
        print(features.shape)

        features = tf.nn.tanh(features)
        print("features.shape")
        print(features.shape)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        print("scores.shape")
        print(scores.shape)
        return scores

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.num_hiddens))
        return x


def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
    dims>, <query attention dims>, num_heads, channels)`

    Args:
        rank: Rank of query, key, value tensors.
        attn_axes: List/tuple of axes, `[-1, rank)`,
            that attention will be applied to.

    Returns:
        Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join(
        [target_notation[i] for i in batch_dims]
        + [target_notation[i] for i in attn_axes]
        + [source_notation[i] for i in attn_axes]
    )
    dot_product_equation = "%s,%s->%s" % (
        source_notation,
        target_notation,
        product_notation,
    )
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (
        product_notation,
        source_notation,
        target_notation,
    )
    return dot_product_equation, combine_equation, attn_scores_rank


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


class MultiHeadAttention(Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.

    When using `MultiHeadAttention` inside a custom layer, the custom layer must
    implement its own `build()` method and call `MultiHeadAttention`'s
    `_build_from_signature()` there.
    This enables weights to be restored correctly when the model is loaded.

    Examples:

    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

    >>> layer = MultiHeadAttention(
    ...     num_heads=2, key_dim=2, attention_axes=(2, 3))
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer(input_tensor, input_tensor)
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        value_dim: Size of each attention head for value.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        output_shape: The expected shape of an output tensor, besides the batch
            and sequence dims. If not specified, projects back to the key
            feature dim.
        attention_axes: axes over which the attention is applied. `None` means
            attention over all axes, but batch, heads, and features.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.

    Call arguments:
        query: Query `Tensor` of shape `(B, T, dim)`.
        value: Value `Tensor` of shape `(B, S, dim)`.
        key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output should
            be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Defaults to either using the training mode of the parent
            layer/model, or False (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: The result of the computation, of shape `(B, T, E)`,
            where `T` is for target sequence shapes and `E` is the query input
            last dimension if `output_shape` is `None`. Otherwise, the
            multi-head outputs are projected to the shape specified by
            `output_shape`.
        attention_scores: [Optional] multi-head attention coefficients over
            attention axes.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        self.W_q = Dense(self._key_dim)  # Learned projection matrix for the queries
        self.W_k = Dense(self._key_dim)  # Learned projection matrix for the keys
        self.W_v = Dense(self._value_dim)  # Learned projection matrix for the values

        if attention_axes is not None and not isinstance(
            attention_axes, collections.abc.Sized
        ):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        self._query_shape, self._key_shape, self._value_shape = None, None, None

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "attention_axes": self._attention_axes,
            "kernel_initializer": initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self._kernel_constraint),
            "bias_constraint": constraints.serialize(self._bias_constraint),
            "query_shape": self._query_shape,
            "key_shape": self._key_shape,
            "value_shape": self._value_shape,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        # If the layer has a different build() function from the Keras default,
        # we need to trigger the customized build to create weights.
        query_shape = config.pop("query_shape")
        key_shape = config.pop("key_shape")
        value_shape = config.pop("value_shape")
        layer = cls(**config)
        if None in [query_shape, key_shape, value_shape]:
            logging.warning(
                "One of dimensions of the input shape is missing. It "
                "should have been memorized when the layer was serialized. "
                "%s is created without weights.",
                str(cls),
            )
        else:
            layer._build_from_signature(query_shape, value_shape, key_shape)
        return layer

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

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def _make_output_dense(self, free_dims, common_kwargs, name=None):
        """Builds the output projection matrix.

        Args:
            free_dims: Number of free dimensions for einsum equation building.
            common_kwargs: Common keyword arguments for einsum layer.
            name: Name for the projection layer.

        Returns:
            Projection layer.
        """
        if self._output_shape:
            if not isinstance(self._output_shape, collections.abc.Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [self._query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=2, output_dims=len(output_shape)
        )
        return core.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **common_kwargs,
        )

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
            rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes), attn_scores_rank
            )
        )
        self._softmax = activation.Softmax(axis=norm_axes)
        self._dropout_layer = regularization.Dropout(rate=self._dropout)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # attention_scores = tf.einsum(self._dot_product_equation, key, query)

        self.attention = DotProductAttention(self.W_q, self.W_k, self.W_v, self._num_heads, self._key_dim)
        attention_scores = self.attention(query, key, value, self._key_dim)
        #
        # self.attention = AdditiveAttention(self._key_dim)
        # attention_scores = self.attention(query, key)

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        print("\n")
        print("attention_scores_dropout.shape")
        print(attention_scores_dropout.shape)
        print("value.shape")
        print(value.shape)
        print("\n")

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        # Convert RaggedTensor to Tensor.
        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()
        key_is_ragged = isinstance(key, tf.RaggedTensor)
        value_is_ragged = isinstance(value, tf.RaggedTensor)
        if key_is_ragged and value_is_ragged:
            # Ensure they have the same shape.
            bounding_shape = tf.math.maximum(
                key.bounding_shape(), value.bounding_shape()
            )
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))

        attention_mask = self._compute_attention_mask(
            query,
            value,
            key=key,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _compute_attention_mask(
        self, query, value, key=None, attention_mask=None, use_causal_mask=False
    ):
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
        query_mask = getattr(query, "_keras_mask", None)
        value_mask = getattr(value, "_keras_mask", None)
        key_mask = getattr(key, "_keras_mask", None)
        auto_mask = None
        if query_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
            # B = batch size, T = max query length
            auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
            # B = batch size, S == max value length
            mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else tf.cast(attention_mask, bool) & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean `Tensor` equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query `Tensor` of shape `(B, T, ...)`.
            value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean `Tensor` of shape [1, T, S] containing a lower
                triangular matrix of shape [T, S].
        """
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )

    def compute_output_shape(self, query_shape, value_shape, key_shape=None):

        if key_shape is None:
            key_shape = value_shape

        query_shape = tf.TensorShape(query_shape)
        value_shape = tf.TensorShape(value_shape)
        key_shape = tf.TensorShape(key_shape)

        if query_shape[-1] != value_shape[-1]:
            raise ValueError(
                "The last dimension of `query_shape` and `value_shape` "
                f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
                "Received: query_shape={query_shape}, value_shape={value_shape}"
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received {value_shape} and "
                f"{key_shape}"
            )

        if self._output_shape:
            return query_shape[:-1].concatenate(self._output_shape)

        return query_shape