#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from config.definitions import ROOT_DIR
import numpy as np


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_dropout = layers.Dropout(rate=dropout)
        self.attention_norm = layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = layers.Dense(units=ff_dim, activation="relu")
        self.dense2 = layers.Dense(units=embed_dim)
        self.dropout1 = layers.Dropout(rate=dropout)
        self.dropout2 = layers.Dropout(rate=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        # Multi-Head Attention block
        attention_output = self.attention(inputs, inputs)
        attention_output = self.attention_dropout(attention_output, training=training)
        attention_output = self.attention_norm(inputs + attention_output)

        # Feed Forward block
        dense_output = self.dense1(attention_output)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout1(dense_output, training=training)
        dense_output = self.dropout2(dense_output, training=training)
        encoder_output = self.norm(attention_output + dense_output)

        return encoder_output


def get_transformer_encoder_model(sequence_length, num_features, num_classes):
    inputs = layers.Input(shape=(sequence_length, num_features))

    # Additive Attention
    query = layers.Dense(units=sequence_length)(inputs)
    key = layers.Dense(units=sequence_length)(inputs)
    value = layers.Dense(units=num_features)(inputs)
    attention_scores = layers.Dot(axes=-1)([query, key])
    attention_scores = layers.Activation('softmax')(attention_scores)
    attention_output = layers.Dot(axes=1)([attention_scores, value])

    # Transformer Encoder block
    transformer_output = TransformerEncoder(embed_dim=num_features, num_heads=4, ff_dim=64)(attention_output)
    transformer_output = TransformerEncoder(embed_dim=num_features, num_heads=4, ff_dim=64)(transformer_output)

    flatten_output = layers.Flatten()(transformer_output)
    dense_output = layers.Dense(units=num_classes, activation="softmax")(flatten_output)

    model = keras.models.Model(inputs=inputs, outputs=dense_output)

    return model


if __name__ == '__main__':
    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150
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

    # model = build_model(input_shape, head_size=16, num_heads=4, ff_dim=1, num_transformer_blocks=1, n_classes=n_labels,
    #                     mlp_units=[2])
    # , )

    model = get_transformer_encoder_model(sequence_length=20, num_features=12, num_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #               metrics=["sparse_categorical_accuracy"])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file="tst_model.png", show_shapes=True)

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
   
    # model.evaluate(x_test, y_test, verbose=1)

    test_data = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_test_data_20ms.npy")

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    x_test = np.reshape(x_test, (test_data.shape[0], 20, 13, 1))
    y_test = to_categorical(y_test)

    x_test = x_test[:, :, 1:, :]

    predictions_list = []

    for i in range(0, len(test_data)):
        prediction = model.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
        # Reverse to_categorical from keras utils
        decoded_prediction = np.argmax(prediction, axis=1, out=None)
        predictions_list.append(decoded_prediction)

    predicted_values = np.asarray(predictions_list)

    # Reverse to_categorical from keras utils
    # predicted_values = np.argmax(predicted_values, axis=1, out=None)

    true_values = test_data[:, -1]

    cm = confusion_matrix(y_true=true_values, y_pred=predicted_values)
    print(cm)

