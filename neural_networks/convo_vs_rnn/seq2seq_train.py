#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from config.definitions import ROOT_DIR
import numpy as np
from neural_networks.utils import training_encoder_decoder

if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 25
    time_steps = 50
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    # ADJUSTED NORMALIZATION DATA --------------------------------------------------------------------------------
    # sorted_data_for_learning = SortedDataForLearning(
    #     path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")
    # training_data = sorted_data_for_learning.training_data
    # -------------------------------------------------------------------------------------------------------------

    # UNIVERSAL NORMALIZATION DATA ------------------------------------------------------------------------------
    training_data = np.load(ROOT_DIR + "/data_storage/data/universal_norm/normalized_data.npy")
    # -------------------------------------------------------------------------------------------------------------

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split

    x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0]), time_steps, 13))
    y_train = to_categorical(training_data[:, -1])
    x_train = x_train[:, :, 1:]

    y_train_final = np.zeros((len(training_data), time_steps, y_train.shape[1]))
    for line in range(0, y_train.shape[0]):
        result = y_train[line, :]
        for t in range(0, time_steps):
            y_train_final[line][t] = result

    model = training_encoder_decoder(out_dim=neurons, input_params=params, out_labels=labels, start_n=start_number,
                                     batch_s=batch_size, time_ss=time_steps)
    # model_encoder_decoder_Bahdanau_Attention.summary()

    x_train = x_train[0:2500]
    y_train_final = y_train_final[0:2500]

    print("x_train.shape")
    print(x_train.shape)
    print("y_train_final.shape")
    print(y_train_final.shape)

    callback = EarlyStopping(monitor='val_loss', patience=10)
    fit_history = model.fit(x_train, y_train_final, batch_size=batch_size, epochs=epochs,
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
    # plt.show()

    model.save("RNN_LSTM_attention_50ts_universal_norm")

    plt.savefig(ROOT_DIR + "/neural_networks/convo_vs_rnn/training_curves.png", bbox_inches='tight')

