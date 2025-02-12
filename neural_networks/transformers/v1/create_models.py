#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder


def create_transformer_v1_0():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers = 1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_0"


def create_transformer_v1_1():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=8, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_1"


def create_transformer_v1_2():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_2"


def create_transformer_v1_3():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=8, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_3"


def create_transformer_v1_4():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=2
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_4"


def create_transformer_v1_5():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=2
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=8, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_5"