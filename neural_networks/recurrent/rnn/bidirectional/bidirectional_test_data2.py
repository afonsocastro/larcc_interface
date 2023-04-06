#!/usr/bin/env python3
#
import keras
from numpy import save
from progressbar import progressbar

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np


if __name__ == '__main__':

    sliding_window = 20
    time_steps = 100
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    sorted_data_for_learning = SortedDataForLearning(path=path)

    test_data = sorted_data_for_learning.test_data
    n_test = test_data.shape[0]

    bidirectional_model = keras.models.load_model("bidirectional_20ms")

    x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
    x_test = x_test[:, :, 1:]
    y_test = test_data[:, -1]

    # BIDIRECTIONAL TESTING
    pred_bidirectional = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = bidirectional_model.predict(x=x_test[i:i + 1, sw:sw + sliding_window, :], verbose=2)
            sample_pred.append(prediction)

        pred_bidirectional.append(sample_pred)

    pred_bidirectional = np.array(pred_bidirectional)
    print("\n")
    print("pred_bidirectional.shape")
    print(pred_bidirectional.shape)
    print("\n")
    pred_bidirectional = np.reshape(pred_bidirectional, (
    pred_bidirectional.shape[0], pred_bidirectional.shape[1], pred_bidirectional.shape[3]))

    save('bidirectional_20ms_data2_pred.npy', pred_bidirectional)
