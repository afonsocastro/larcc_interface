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

    # x = np.array([2, 3, 4, 5, 6])
    # y = np.array([0, 1, 2, 3, 4])
    # f = InterpolatedUnivariateSpline(x, y, k=1)  # k=1 gives linear interpolation
    # print("f.integral(1.5, 2.2)")
    # print(f.integral(3, 6))

    pred_cnn_data = load('../convolutional/cnn4_20ms/cnn4_20ms_data3_pred.npy')
    pred_seq2label_data = load('../recurrent/rnn/seq2label/seq2label_20ms_data3_pred.npy')
    pred_transformer_data = load('../transformers/time-series/ts_transformer_data3_pred.npy')
    all_true = load('../convolutional/cnn4_20ms/true_results_data3.npy')

    times = np.array([i for i in range(0, time_steps)])
    area_pull_cnn, area_push_cnn, area_shake_cnn, area_twist_cnn = 0, 0, 0, 0
    area_pull_seq2label, area_push_seq2label, area_shake_seq2label, area_twist_seq2label = 0, 0, 0, 0
    area_pull_transformer, area_push_transformer, area_shake_transformer, area_twist_transformer = 0, 0, 0, 0

    for n in range(0, len(all_true)):

        true = all_true[n]
        pred_cnn = pred_cnn_data[n]
        pred_seq2label = pred_seq2label_data[n]
        pred_transformer = pred_transformer_data[n]

        pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(pred_cnn, time_steps-sliding_window+1)
        pull_seq2label, push_seq2label, shake_seq2label, twist_seq2label = value_for_array(pred_seq2label, time_steps-sliding_window+1)
        pull_transformer, push_transformer, shake_transformer, twist_transformer = value_for_array(pred_transformer, time_steps-sliding_window+1)

        labels = []
        start_number = 0
        for i in times:
            if (i != 0 and true[i] != true[i - 1]) or i == 5999:
                if i == 5999:
                    labels.append({"label": true[start_number], "x": [start_number, 6000]})
                else:
                    labels.append({"label": true[start_number], "x": [start_number, i]})
                start_number = i

        for primitive in labels:

            if primitive["x"][0] == 0:
                xt = np.array([i for i in range(19, primitive["x"][1])])
            else:
                xt = np.array([i for i in range(primitive["x"][0], primitive["x"][1])])

            if primitive["label"] == 0:

                f_cnn = InterpolatedUnivariateSpline(xt, pull_cnn[xt[0]-19: xt[-1]-18], k=1)
                area_pull_cnn += f_cnn.integral(xt[0], xt[-1])

                f_seq2label = InterpolatedUnivariateSpline(xt, pull_seq2label[xt[0] - 19: xt[-1] - 18], k=1)
                area_pull_seq2label += f_seq2label.integral(xt[0], xt[-1])

                f_transformer = InterpolatedUnivariateSpline(xt, pull_transformer[xt[0] - 19: xt[-1] - 18], k=1)
                area_pull_transformer += f_transformer.integral(xt[0], xt[-1])

            elif primitive["label"] == 1:

                f_cnn = InterpolatedUnivariateSpline(xt, push_cnn[xt[0] - 19: xt[-1] - 18], k=1)
                area_push_cnn += f_cnn.integral(xt[0], xt[-1])

                f_seq2label = InterpolatedUnivariateSpline(xt, push_seq2label[xt[0] - 19: xt[-1] - 18], k=1)
                area_push_seq2label += f_seq2label.integral(xt[0], xt[-1])

                f_transformer = InterpolatedUnivariateSpline(xt, push_transformer[xt[0] - 19: xt[-1] - 18], k=1)
                area_push_transformer += f_transformer.integral(xt[0], xt[-1])

            elif primitive["label"] == 2:

                f_cnn = InterpolatedUnivariateSpline(xt, shake_cnn[xt[0] - 19: xt[-1] - 18], k=1)
                area_shake_cnn += f_cnn.integral(xt[0], xt[-1])

                f_seq2label = InterpolatedUnivariateSpline(xt, shake_seq2label[xt[0] - 19: xt[-1] - 18], k=1)
                area_shake_seq2label += f_seq2label.integral(xt[0], xt[-1])

                f_transformer = InterpolatedUnivariateSpline(xt, shake_transformer[xt[0] - 19: xt[-1] - 18], k=1)
                area_shake_transformer += f_transformer.integral(xt[0], xt[-1])

            elif primitive["label"] == 3:
                f_cnn = InterpolatedUnivariateSpline(xt, twist_cnn[xt[0] - 19: xt[-1] - 18], k=1)
                area_twist_cnn += f_cnn.integral(xt[0], xt[-1])

                f_seq2label = InterpolatedUnivariateSpline(xt, twist_seq2label[xt[0] - 19: xt[-1] - 18], k=1)
                area_twist_seq2label += f_seq2label.integral(xt[0], xt[-1])

                f_transformer = InterpolatedUnivariateSpline(xt, twist_transformer[xt[0] - 19: xt[-1] - 18], k=1)
                area_twist_transformer += f_transformer.integral(xt[0], xt[-1])

    # print("area_pull_cnn")
    # print(area_pull_cnn)
    # print("\n")
    #
    # print("area_push_cnn")
    # print(area_push_cnn)
    # print("\n")
    #
    # print("area_shake_cnn")
    # print(area_shake_cnn)
    # print("\n")
    #
    # print("area_twist_cnn")
    # print(area_twist_cnn)
    # print("\n")
    #
    # print("area_pull_seq2label")
    # print(area_pull_seq2label)
    # print("\n")
    #
    # print("area_push_seq2label")
    # print(area_push_seq2label)
    # print("\n")
    #
    # print("area_shake_seq2label")
    # print(area_shake_seq2label)
    # print("\n")
    #
    # print("area_twist_seq2label")
    # print(area_twist_seq2label)
    # print("\n")
    #
    # print("area_pull_transformer")
    # print(area_pull_transformer)
    # print("\n")
    #
    # print("area_push_transformer")
    # print(area_push_transformer)
    # print("\n")
    #
    # print("area_shake_transformer")
    # print(area_shake_transformer)
    # print("\n")
    #
    # print("area_twist_transformer")
    # print(area_twist_transformer)
    # print("\n")

    total_area_cnn = area_pull_cnn + area_push_cnn + area_shake_cnn + area_twist_cnn
    total_area_seq2label = area_pull_seq2label + area_push_seq2label + area_shake_seq2label + area_twist_seq2label
    total_area_transformer = area_pull_transformer + area_push_transformer + area_shake_transformer + area_twist_transformer

    print("total_area_cnn")
    print(total_area_cnn)
    print("total_area_seq2label")
    print(total_area_seq2label)
    print("total_area_transformer")
    print(total_area_transformer)
