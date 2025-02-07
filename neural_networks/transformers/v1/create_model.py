#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder

from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from sklearn.metrics import confusion_matrix

from larcc_interface.larcc_classes.documentation.PDF import PDF
from larcc_interface.neural_networks.utils import plot_confusion_matrix_percentage, prediction_classification, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay

from larcc_interface.config.definitions import ROOT_DIR
import numpy as np

def create_transformer_1(input_shape, num_layers, num_heads):
    inputs = keras.Input(shape=input_shape)
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding

    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=num_heads, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model

if __name__ == '__main__':

    time_steps = 20
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    validation_split = 0.3

    training_data = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_train_data_20ms.npy")
    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], time_steps, 13))
    y_train = to_categorical(training_data[:, -1])

    x_train = x_train[:, :, 1:]
    # x_train = training_data[:, :-1]
    print("x_train.shape")
    print(x_train.shape)
    print("y_train.shape")
    print(y_train.shape)
    print("y_train")
    print(y_train)

    input_shp = x_train.shape[1:]

    print("input_shape")
    print(input_shp)

    model = create_transformer_1(input_shp, num_layers=2, num_heads=8)

    # loss = "sparse_categorical_crossentropy", optimizer = keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file="transformer_v1_1.png", show_shapes=True)

    # callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)]

    fit_history = model.fit(x_train, y_train, shuffle=True, validation_split=validation_split, epochs=500, batch_size=32)
                            # callbacks=callbacks)
    model.save("transformer_v1_1.keras")
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