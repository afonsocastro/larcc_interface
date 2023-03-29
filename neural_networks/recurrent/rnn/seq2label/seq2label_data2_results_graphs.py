#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def value_for_array(data, n, timesteps):
    pull = np.array([data[n][j][0] for j in range(timesteps)])
    push = np.array([data[n][j][1] for j in range(timesteps)])
    shake = np.array([data[n][j][2] for j in range(timesteps)])
    twist = np.array([data[n][j][3] for j in range(timesteps)])

    return pull, push, shake, twist


def string_result(true):
    results = []

    for i in range(0, 2):
        if true[i] == 0:
            result = "PULL"
        elif true[i] == 1:
            result = "PUSH"
        elif true[i] == 2:
            result = "SHAKE"
        elif true[i] == 3:
            result = "TWIST"
        results.append(result)

    return results


if __name__ == '__main__':
    time_steps = 100
    sliding_window = 20

    pred_seq2label = load('seq2label_20ms_data2_pred.npy')
    true = load('../true_results_data2.npy')

    window_times = np.array([i for i in range(19, 100)])

    for n in range(0, len(pred_seq2label)):

        fig = plt.figure(figsize=(12, 8))

        # CONVOLUTIONAL GRAPH
        pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(pred_seq2label, n, time_steps-sliding_window+1)

        df11 = pd.DataFrame({'timestep': window_times, 'pull_cnn': pull_cnn})
        df22 = pd.DataFrame({'timestep': window_times, 'push_cnn': push_cnn})
        df33 = pd.DataFrame({'timestep': window_times, 'shake_cnn': shake_cnn})
        df44 = pd.DataFrame({'timestep': window_times, 'twist_cnn': twist_cnn})

        plt.plot(df11.timestep, df11.pull_cnn, color="blue", label='pull', linewidth=3)
        plt.plot(df22.timestep, df22.push_cnn, color='red', label='push', linewidth=3)
        plt.plot(df33.timestep, df33.shake_cnn, color='green', label='shake', linewidth=3)
        plt.plot(df44.timestep, df44.twist_cnn, color='orange', label='twist', linewidth=3)

        # Add the vertical line to the plot
        plt.axvline(x=50, linestyle="--")

        results_true = string_result(true[n])
        # plt.title('CNN output confidences at timestep. True Expected: ' + results_true[0] + ' and ' + results_true[1])
        plt.title("Ground Truth:    " + results_true[0] + '  -->  ' + results_true[1],
                  fontdict={"fontsize": 16, "fontweight": "bold"})
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()

