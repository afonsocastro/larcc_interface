#!/usr/bin/env python3

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
from neural_networks.utils import training_encoder_decoder

if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 2
    # time_steps = 50
    time_steps = 100
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    sorted_data_for_learning = SortedDataForLearning(
        path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    # n_test = test_data.shape[0]

    x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0] / 2), time_steps, 13))
    # x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0]), time_steps, 13))

    y_train = to_categorical(training_data[:, -1])
    x_train = x_train[:, :, 1:]

    results = []
    y_train_final = []
    for line in range(0, y_train.shape[0], 2):
        result1 = y_train[line, :]
        result2 = y_train[line+1, :]

        for i in range(0, int(x_train.shape[1] / 2)):
            results.append(result1)
        for i in range(int(x_train.shape[1] / 2), x_train.shape[1]):
            results.append(result2)

        y_train_final.append(results)
        results = []
    y_train_final = np.array(y_train_final, dtype=float)

    model_encoder_decoder_Bahdanau_Attention = training_encoder_decoder(neurons, params, labels, start_number, batch_size, time_steps)
    # model_encoder_decoder_Bahdanau_Attention.summary()

    # plot_model(model_encoder_decoder_Bahdanau_Attention, to_file="seq2seq/model.png", show_shapes=True)

    print("x_train.shape")
    print(x_train.shape)
    print("y_train_final.shape")
    print(y_train_final.shape)

    callback = EarlyStopping(monitor='val_loss', patience=10)
    fit_history = model_encoder_decoder_Bahdanau_Attention.fit(x_train, y_train_final, batch_size=batch_size,
                                                               epochs=epochs, validation_split=validation_split,
                                                               shuffle=True, verbose=2, callbacks=[callback])

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

    model_encoder_decoder_Bahdanau_Attention.save("just_for_fun")
    exit(0)

    # plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq/training_curves_RNN_adjusted_norm.png", bbox_inches='tight')

    # model_encoder_decoder_Bahdanau_Attention.save("model_Bahdanau_Attention")

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

    pred = model_encoder_decoder_Bahdanau_Attention.predict(x_test[0].reshape(1, x_test[0].shape[0], x_test[0].shape[1]))

    print('input: ', x_test[0])
    print('expected: ', y_test_final[0])
    print('predicted: ', pred[0])

    exit(0)

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
