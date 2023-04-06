#!/usr/bin/env python3

import keras
import numpy as np
from numpy import argmax, array_equal, save
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
from config.definitions import ROOT_DIR
from progressbar import progressbar


def predict_sequence(infenc, infdec, source, n_steps, cardinality, sn):
    # encode
    if len(source.shape) == 2:
        source = np.reshape(source, (1, source.shape[0], source.shape[1]))
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([sn for _ in range(cardinality)])
    target_seq = np.reshape(target_seq, (1, 1, cardinality))
    # collect predictions
    output = list()
    for t in range(n_steps):
    # for _ in progressbar(range(n_steps), redirect_stdout=True):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat

    return np.array(output)


if __name__ == '__main__':
    sliding_window = 20
    time_steps = 6000
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    cnn_model = keras.models.load_model("cnn4_model_20ms")

    test_data = np.load(ROOT_DIR + "/data_storage/data3/global_normalized_data.npy")
    n_test = test_data.shape[0]

    x_test_cnn = np.reshape(test_data[:, :, :-1], (int(n_test), time_steps, 13, 1))
    x_test_cnn = x_test_cnn[:, :, 1:, :]
    y_test = test_data[:, :, -1]

    # CONVOLUTIONAL TESTING
    pred_cnn = []
    for i in progressbar(range(len(x_test_cnn)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = cnn_model.predict(x=x_test_cnn[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred_cnn.append(sample_pred)

    pred_cnn = np.array(pred_cnn)
    print("\n")
    print("pred_cnn.shape")
    print(pred_cnn.shape)
    print("\n")
    pred_cnn = np.reshape(pred_cnn, (pred_cnn.shape[0], pred_cnn.shape[1], pred_cnn.shape[3]))

    save('cnn4_20ms_data3_pred.npy', pred_cnn)
    save('true_results_data3.npy', y_test)
