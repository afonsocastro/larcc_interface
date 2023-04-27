import keras
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical, plot_model
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Embedding, Flatten
from tensorflow.keras.models import Model

from config.definitions import ROOT_DIR


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Embedding, Flatten
from tensorflow.keras.models import Model

# Define the transformer block as a layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


# Define the transformer model
def create_transformer_model(num_layers, embed_dim, num_heads, ff_dim, input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Embedding(input_dim=num_classes, output_dim=embed_dim)(inputs)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embeddings = Embedding(input_dim=input_shape[0], output_dim=embed_dim)(positions)
    x = x + position_embeddings
    for i in range(num_layers):
        x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
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

    # print("input_shape")
    # print(input_shape)
    # exit(0)

    # Set the hyperparameters
    num_layers = 1
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    dropout_rate = 0.1

    model = create_transformer_model(num_layers, embed_dim, num_heads, ff_dim, input_shape, n_labels)


    # model = build_model(input_shape, head_size=16, num_heads=4, ff_dim=1, num_transformer_blocks=1, n_classes=n_labels,
    #                     mlp_units=[2])
    # , )

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
