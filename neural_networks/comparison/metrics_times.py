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
    time_steps = 6000
    sliding_window = 20

    pred_cnn_data = load('../convolutional/cnn4_20ms/cnn4_20ms_data3_pred.npy')
    pred_seq2label_data = load('../recurrent/rnn/seq2label/seq2label_20ms_data3_pred.npy')
    pred_transformer_data = load('../transformers/time-series/ts_transformer_data3_pred.npy')
    all_true = load('../convolutional/cnn4_20ms/true_results_data3.npy')

    times = np.array([i for i in range(0, time_steps)])
    total_pull_cnn, total_push_cnn, total_shake_cnn, total_twist_cnn = 0, 0, 0, 0
    total_pull_seq2label, total_push_seq2label, total_shake_seq2label, total_twist_seq2label = 0, 0, 0, 0
    total_pull_transformer, total_push_transformer, total_shake_transformer, total_twist_transformer = 0, 0, 0, 0

    for n in range(0, len(all_true)):

        true = all_true[n]
        pred_cnn = pred_cnn_data[n]
        pred_seq2label = pred_seq2label_data[n]
        pred_transformer = pred_transformer_data[n]

        pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(pred_cnn, time_steps-sliding_window+1)
        pull_seq2label, push_seq2label, shake_seq2label, twist_seq2label = value_for_array(pred_seq2label, time_steps-sliding_window+1)
        pull_transformer, push_transformer, shake_transformer, twist_transformer = value_for_array(pred_transformer, time_steps-sliding_window+1)

        # labels = []
        # start_number = 0
        for i in times:
            if i >= 19:
                pred_cnn_i = np.array([pull_cnn[i-19], push_cnn[i-19], shake_cnn[i-19], twist_cnn[i-19]]).argmax()
                pred_seq2label_i = np.array([pull_seq2label[i-19], push_seq2label[i-19], shake_seq2label[i-19], twist_seq2label[i-19]]).argmax()
                pred_transformer_i = np.array([pull_transformer[i-19], push_transformer[i-19], shake_transformer[i-19], twist_transformer[i-19]]).argmax()

                if true[i] == 0:
                    if pred_cnn_i == 0:
                        total_pull_cnn += 1
                    if pred_seq2label_i == 0:
                        total_pull_seq2label += 1
                    if pred_transformer_i == 0:
                        total_pull_transformer += 1

                elif true[i] == 1:
                    if pred_cnn_i == 1:
                        total_push_cnn += 1
                    if pred_seq2label_i == 1:
                        total_push_seq2label += 1
                    if pred_transformer_i == 1:
                        total_push_transformer += 1

                elif true[i] == 2:
                    if pred_cnn_i == 2:
                        total_shake_cnn += 1
                    if pred_seq2label_i == 2:
                        total_shake_seq2label += 1
                    if pred_transformer_i == 2:
                        total_shake_transformer += 1

                elif true[i] == 3:
                    if pred_cnn_i == 3:
                        total_twist_cnn += 1
                    if pred_seq2label_i == 3:
                        total_twist_seq2label += 1
                    if pred_transformer_i == 3:
                        total_twist_transformer += 1

    print("total_pull_cnn")
    print(total_pull_cnn)
    print("\n")

    print("total_push_cnn")
    print(total_push_cnn)
    print("\n")

    print("total_shake_cnn")
    print(total_shake_cnn)
    print("\n")

    print("total_twist_cnn")
    print(total_twist_cnn)
    print("\n")

    print("total_pull_seq2label")
    print(total_pull_seq2label)
    print("\n")

    print("total_push_seq2label")
    print(total_push_seq2label)
    print("\n")

    print("total_shake_seq2label")
    print(total_shake_seq2label)
    print("\n")

    print("total_twist_seq2label")
    print(total_twist_seq2label)
    print("\n")

    print("total_pull_transformer")
    print(total_pull_transformer)
    print("\n")

    print("total_push_transformer")
    print(total_push_transformer)
    print("\n")

    print("total_shake_transformer")
    print(total_shake_transformer)
    print("\n")

    print("total_twist_transformer")
    print(total_twist_transformer)
    print("\n")


    total_times_cnn = total_pull_cnn + total_push_cnn + total_shake_cnn + total_twist_cnn
    total_times_seq2label = total_pull_seq2label + total_push_seq2label + total_shake_seq2label + total_twist_seq2label
    total_times_transformer = total_pull_transformer + total_push_transformer + total_shake_transformer + total_twist_transformer

    print("total_times_cnn")
    print(total_times_cnn)
    print("total_times_seq2label")
    print(total_times_seq2label)
    print("total_times_transformer")
    print(total_times_transformer)
