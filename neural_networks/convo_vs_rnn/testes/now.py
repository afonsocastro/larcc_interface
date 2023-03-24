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
    validation_split = 0.3
    batch_size = 25
    # time_steps_in = 50
    # time_steps_out = 6000
    time_steps_out = 50
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    infenc = keras.models.load_model("infenc_model")
    infdec = keras.models.load_model("infdec_model")

    # sorted_data_for_learning = SortedDataForLearning(
    #     path=ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/")
    # test_data = sorted_data_for_learning.test_data
    #
    # x_test = np.reshape(test_data[:, :-1], (int(test_data.shape[0]), time_steps_in, 13))
    # x_test = x_test[:, :, 1:]
    #
    # print("x_test.shape")
    # print(x_test.shape)
    # print("x_test[0]")
    # print(x_test[0])

    test_data = np.load(ROOT_DIR + "/data_storage/full_timewindow/data/universal_normalized_data.npy")
    all_predicts = []
    all_trues = []

    # for n in range(0, 3):
    n_data = test_data.shape[0]
    for n in progressbar(range(0, 2), redirect_stdout=True):
    # for n in range(0, 1):
        one_sample_test = test_data[n]
        y_true = one_sample_test[:, -1]
        one_sample_test = one_sample_test[:, 1:-1]

        # target = predict_sequence(infenc, infdec, one_sample_test, time_steps_out, labels, start_number)
        # # predicted = np.empty((target.shape[0]))
        # predicted = []
        # correct = 0
        # total = one_sample_test.shape[0]

        # target = predict_sequence(infenc, infdec, one_sample_test, time_steps_out, labels, start_number)
        # print(target)
        # predicted = np.empty((target.shape[0]))

        predicted = []

        correct = 0
        total = one_sample_test.shape[0]
        target = predict_sequence(infenc, infdec, one_sample_test[0:50], time_steps_out, labels, start_number)
        for _ in range(0, len(target)):
            predicted.append(target[_])

        for i in range(50, y_true.shape[0]):
        # for i in progressbar(range(50, y_true.shape[0]), redirect_stdout=True):
        # for i in progressbar(range(50, y_true.shape[0], 25), redirect_stdout=True):
            target = predict_sequence(infenc, infdec, one_sample_test[i-49:i+1], time_steps_out, labels, start_number)
            # for t in range(40, 50):
            predicted.append(target[-1])

        # for i in range(0, target.shape[0]):
        #     if int(y_true[i]) == int(argmax(target[i])):
        #         correct += 1
        # print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))

        all_predicts.append(predicted)
        all_trues.append(y_true)

    # print(predicted.shape)
    # print(y_true.shape)
    #
    # # exit(0)
    # for _ in range(0, predicted.shape[0]):
    #     if int(y_true[_]) == int(argmax(predicted[_])):
    #         correct += 1
    #
    # print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
    # all_predicts.append(predicted)
    # all_trues.append(y_true)

    # predicted = np.array(predicted, dtype=float)
    all_predicts = np.array(all_predicts, dtype=float)
    all_trues = np.array(all_trues, dtype=float)
    save("1_results/y_true.npy", all_trues)
    save("1_results/predicted.npy", all_predicts)

    # total, correct = 261, 0
    # # for _ in range(total):
    # for _ in progressbar(range(total), redirect_stdout=True):
    #     target = predict_sequence(infenc, infdec, x_test[_], time_steps_out, labels, start_number)
    #     y_true = test_data[_, -1]
    #     if int(argmax(target[25])) == int(y_true):
    #         correct += 1
    # print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
