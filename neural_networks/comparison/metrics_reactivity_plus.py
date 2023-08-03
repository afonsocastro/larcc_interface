#!/usr/bin/env python3
from time import sleep

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

    return pull, push, shake, twist


if __name__ == '__main__':
    time_steps = 100
    sliding_window = 20

    pred_cnn_data = load('../convolutional/cnn4_20ms/cnn4_20ms_data2_pred.npy')
    pred_seq2label_data = load('../recurrent/rnn/seq2label/seq2label_20ms_data2_pred.npy')
    pred_transformer_data = load('../transformers/time-series/ts_transformer_data2_pred.npy')
    all_true = load('../convolutional/cnn4_20ms/true_results_data2.npy')

    times = np.array([i for i in range(0, time_steps)])
    total_pull_cnn, total_push_cnn, total_shake_cnn, total_twist_cnn = 0, 0, 0, 0
    total_pull_seq2label, total_push_seq2label, total_shake_seq2label, total_twist_seq2label = 0, 0, 0, 0
    total_pull_transformer, total_push_transformer, total_shake_transformer, total_twist_transformer = 0, 0, 0, 0

    total_count_cnn = 0
    total_count_seq2label = 0
    total_count_transformer = 0
    for n in range(0, int(len(all_true)/2)):

        old_true = all_true[n]
        print("old_true.shape")
        print(old_true.shape)
        print("old_true")
        print(old_true)

        true = np.zeros(100)
        for i in range(0, 50):
            true[i] = old_true[0]
        for i in range(50, 100):
            true[i] = old_true[1]

        print("true.shape")
        print(true.shape)
        print("true")
        print(true)

        pred_cnn = pred_cnn_data[n]
        pred_seq2label = pred_seq2label_data[n]
        pred_transformer = pred_transformer_data[n]

        pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(pred_cnn, time_steps - sliding_window + 1)
        pull_seq2label, push_seq2label, shake_seq2label, twist_seq2label = value_for_array(pred_seq2label,
                                                                                           time_steps - sliding_window + 1)
        pull_transformer, push_transformer, shake_transformer, twist_transformer = value_for_array(pred_transformer,
                                                                                                   time_steps - sliding_window + 1)

        # labels = []
        # start_number = 0
        count_cnn = 0
        count_seq2label = 0
        count_transformer = 0
        changed_cnn = False
        changed_seq2label = False
        changed_transformer = False
        for i in times:
            if i > 18:
                pred_cnn_i = np.array(
                    [pull_cnn[i - 19], push_cnn[i - 19], shake_cnn[i - 19], twist_cnn[i - 19]]).argmax()
                pred_seq2label_i = np.array([pull_seq2label[i - 19], push_seq2label[i - 19], shake_seq2label[i - 19],
                                             twist_seq2label[i - 19]]).argmax()
                pred_transformer_i = np.array(
                    [pull_transformer[i - 19], push_transformer[i - 19], shake_transformer[i - 19],
                     twist_transformer[i - 19]]).argmax()

                if true[i] != true[i - 1]:
                    changed_cnn = True
                    changed_seq2label = True
                    changed_transformer = True

                if changed_cnn:
                    if pred_cnn_i == true[i]:
                        changed_cnn = False
                    else:
                        count_cnn += 1

                if changed_seq2label:
                    if pred_seq2label_i == true[i]:
                        changed_seq2label = False
                    else:
                        count_seq2label += 1

                if changed_transformer:
                    if pred_transformer_i == true[i]:
                        changed_transformer = False
                    else:
                        count_transformer += 1

                old_cnn_pred = pred_cnn_i
                old_seq2label_pred = pred_seq2label_i
                old_transformer_pred = pred_transformer_i

        total_count_cnn += count_cnn
        total_count_seq2label += count_seq2label
        total_count_transformer += count_transformer

    cnn_metric = 2 * total_count_cnn / len(all_true)
    seq2label_metric = 2 * total_count_seq2label / len(all_true)
    transformer_metric = 2 * total_count_transformer / len(all_true)

    print("cnn_metric")
    print(cnn_metric)
    print("seq2label_metric")
    print(seq2label_metric)
    print("transformer_metric")
    print(transformer_metric)
