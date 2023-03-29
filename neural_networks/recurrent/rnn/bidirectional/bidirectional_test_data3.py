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

    bidirectional_model = keras.models.load_model("bidirectional_20ms")

    data = np.load(ROOT_DIR + "/data_storage/data3/sequence_normalized_data.npy")
    n_test = data.shape[0]

    x_test = np.reshape(data[:, :, :-1], (int(n_test), time_steps, 13))
    x_test = x_test[:, :, 1:]
    y_test = data[:, :, -1]

    # BIDIRECTIONAL TESTING
    pred_bidirectional = []
    for i in progressbar(range(len(x_test)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = bidirectional_model.predict(x=x_test[i:i+1, sw:sw+sliding_window, :], verbose=2)
            sample_pred.append(prediction)

        pred_bidirectional.append(sample_pred)

    pred_bidirectional = np.array(pred_bidirectional)
    print("\n")
    print("pred_bidirectional.shape")
    print(pred_bidirectional.shape)
    print("\n")
    pred_bidirectional = np.reshape(pred_bidirectional, (
    pred_bidirectional.shape[0], pred_bidirectional.shape[1], pred_bidirectional.shape[3]))

    save('bidirectional_20ms_data3_pred.npy', pred_bidirectional)
