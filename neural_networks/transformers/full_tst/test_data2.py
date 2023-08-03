#!/usr/bin/env python3

import keras
from numpy import save
from progressbar import progressbar
from config.definitions import ROOT_DIR
import numpy as np


if __name__ == '__main__':

    sliding_window = 20
    time_steps = 100
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    print("time_steps-sliding_window+1")
    print(time_steps-sliding_window+1)

    x_test = np.load(ROOT_DIR + "/data_storage/data2/x_test_global_normalized_data.npy")
    n_test = x_test.shape[0]

    tst_model = keras.models.load_model("transformer_model")

    x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_test_cnn = x_test_cnn[:, :, 1:, :]
    y_test = np.load(ROOT_DIR + "/data_storage/data2/y_test_data.npy")

    # TRUE RESULTS
    y_test_final = []
    for line in y_test:
        r = [line[0], line[1]]
        y_test_final.append(r)
    save('true_results_data2.npy', y_test_final)

    # TST TESTING
    pred_tst = []
    for i in progressbar(range(int(len(x_test_cnn)/2)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = tst_model.predict(x=x_test_cnn[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred_tst.append(sample_pred)

    pred_tst = np.array(pred_tst)
    print("\n")
    print("pred_tst.shape")
    print(pred_tst.shape)
    print("\n")
    pred_tst = np.reshape(pred_tst, (pred_tst.shape[0], pred_tst.shape[1], pred_tst.shape[3]))

    save('ts_transformer_data2_pred.npy', pred_tst)
