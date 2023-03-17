#!/usr/bin/env python3
import json

import keras.models
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np

from numpy import save, delete


if __name__ == '__main__':
    time_steps = 6000
    labels = 4
    start_number = 17
    epochs = 50
    time_window = 2
    input_nn = time_window * 10

    test_data = np.load(ROOT_DIR + "/data_storage/full_timewindow/data/raw_learning_data.npy")
    total_predictions = np.empty((18, 5950, 50, 4))

    # UNIVERSAL NORMALIZATION -------------------------------------------------------------------------------------
    # f = open(ROOT_DIR + '/data_storage/src/clusters_max_min.json')
    # clusters_max_min = json.load(f)
    # f.close()
    #
    # data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    # data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    # data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    # data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    # RNN_universal_norm_model = keras.models.load_model("RNN_LSTM_attention_50ts_universal_norm")

    # for n_sample in range(0, test_data.shape[0]):
    #     sample = test_data[n_sample]
    #     ground_truth = sample[:, -1]
    #     # sample = sample[:, 1:-1]
    #     data_array_norm = np.empty((sample.shape[0], 0))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 0:1] / data_max_timestamp))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 1:7] / data_max_joints))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 7:10] / data_max_gripper_F))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 10:13] / data_max_gripper_M))
    #     data_universal_norm = data_array_norm[:, 1:]
    #
    #     data = np.empty((time_steps - 50, 50, 12))
    #     for ts in range(0, time_steps-51):
    #         data[ts] = data_universal_norm[ts:50+ts].reshape(1, 50, 12)
    #     pred = RNN_universal_norm_model.predict(data, batch_size=25)
    #     total_predictions[n_sample] = pred
    #
    save('rnn_universal_norm_pred.npy', total_predictions)
    # ---------------------------------------------------------------------------------------------------------------

    # ADJUSTABLE NORMALIZATION -------------------------------------------------------------------------------------
    RNN_adjustable_norm_model = keras.models.load_model("RNN_LSTM_attention_50ts_adjustable_norm")
    for n_sample in range(0, test_data.shape[0]):
        sample = test_data[n_sample]
        ground_truth = sample[:, -1]
        sample = sample[:, 1:-1]
        data = np.empty((time_steps - 50, 50, 12))
        for ts in range(0, time_steps - 51):
            s = sample[ts:50 + ts]
            data_array_norm = np.empty((50, 0))
            idx = 0
            for n in [6, 3, 3]:
                data_sub_array = s[:, idx:idx + n]
                idx += n
                data_max = abs(max(data_sub_array.min(), data_sub_array.max(), key=abs))
                data_sub_array_norm = data_sub_array / data_max
                data_array_norm = np.hstack((data_array_norm, data_sub_array_norm))
            data[ts] = data_array_norm[:, :].reshape(1, 50, 12)
        pred = RNN_adjustable_norm_model.predict(data, batch_size=25)
        total_predictions[n_sample] = pred

    save('rnn_adjustable_norm_pred.npy', total_predictions)
    # ---------------------------------------------------------------------------------------------------------------
    exit(0)



    # n_test = test_data.shape[0]
    #
    # cnn4_model = keras.models.load_model(ROOT_DIR + "/neural_networks/convolutional/cnn4_model_02_time_window")
    # Bahdanau_Attention_model = keras.models.load_model("model_Bahdanau_Attention")
    #
    # x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
    # y_test = test_data[:, -1]
    # x_test_rnn = x_test[:, :, 1:]
    # x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    #
    # # CONVOLUTIONAL TESTING
    # pred_cnn = []
    # for i in range(0, len(x_test)-1):
    #     sample_pred = []
    #     for sw in range(0, 81):
    #         prediction = cnn4_model.predict(x=x_test_cnn[i:i+1, sw:sw+20, :, :], verbose=0)
    #         sample_pred.append(prediction)
    #
    #     pred_cnn.append(sample_pred)
    #
    # pred_cnn = np.array(pred_cnn)
    # pred_cnn = np.reshape(pred_cnn, (pred_cnn.shape[0], pred_cnn.shape[1], pred_cnn.shape[3]))
    #
    # save('cnn4_model_pred.npy', pred_cnn)
    #
    # # RECURRENT TESTING
    # pred = Bahdanau_Attention_model.predict(x_test_rnn[0:260].reshape(260, time_steps, params), batch_size=2)
    # save('Bahdanau_Attention_model_pred.npy', pred)
    #
    # # TRUE RESULTS
    # y_test_final = []
    # for line in range(0, y_test.shape[0], 2):
    #     r = [int(y_test[line]), int(y_test[line + 1])]
    #     y_test_final.append(r)
    # save('true_results_cnn4_vs_rnnAttention.npy', y_test_final[0:260])
    #
    #
