#!/usr/bin/env python3
import keras.models
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np

from numpy import save, delete


if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 2
    time_steps = 100
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50
    time_window = 2
    input_nn = time_window * 10

    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    sorted_data_for_learning = SortedDataForLearning(path=path)

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    n_test = test_data.shape[0]

    cnn4_model = keras.models.load_model(ROOT_DIR + "/neural_networks/convolutional/cnn4_model_02_time_window")
    Bahdanau_Attention_model = keras.models.load_model("model_Bahdanau_Attention")

    x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
    y_test = test_data[:, -1]
    x_test_rnn = x_test[:, :, 1:]
    x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # CONVOLUTIONAL TESTING
    pred_cnn = []
    for i in range(0, len(x_test)-1):
        sample_pred = []
        for sw in range(0, 81):
            prediction = cnn4_model.predict(x=x_test_cnn[i:i+1, sw:sw+20, :, :], verbose=0)
            sample_pred.append(prediction)

        pred_cnn.append(sample_pred)

    pred_cnn = np.array(pred_cnn)
    pred_cnn = np.reshape(pred_cnn, (pred_cnn.shape[0], pred_cnn.shape[1], pred_cnn.shape[3]))

    save('cnn4_model_pred.npy', pred_cnn)

    # RECURRENT TESTING
    pred = Bahdanau_Attention_model.predict(x_test_rnn[0:260].reshape(260, time_steps, params), batch_size=2)
    save('Bahdanau_Attention_model_pred.npy', pred)

    # TRUE RESULTS
    y_test_final = []
    for line in range(0, y_test.shape[0], 2):
        r = [int(y_test[line]), int(y_test[line + 1])]
        y_test_final.append(r)
    save('true_results_cnn4_vs_rnnAttention.npy', y_test_final[0:260])


