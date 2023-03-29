#!/usr/bin/env python3

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


def plot_true_shadow(ts, t, p):
    start = 0
    for i in ts:
        if (i != 0 and t[i] != t[i-1]) or i == 5999:
            end = i
            if t[i-1] == 0:
                color = "blue"
            elif t[i-1] == 1:
                color = "red"
            elif t[i-1] == 2:
                color = "green"
            elif t[i-1] == 3:
                color = "orange"
            p.axvspan(start, end, color=color, alpha=0.2, lw=0)
            start = end


if __name__ == '__main__':
    time_steps = 6000
    sliding_window = 20

    pred_seq2label = load('seq2label_20ms_data3_pred.npy')
    all_true = load('../true_results_data3.npy')

    seq2label_times = np.array([i for i in range(19, time_steps)])
    times = np.array([i for i in range(0, time_steps)])

    for n in range(0, len(all_true)):

        true = all_true[n]
        plot_true_shadow(times, true, plt)
        pred = pred_seq2label[n]

        # SEQUENCE 2 LABEL GRAPH
        pull_seq2label, push_seq2label, shake_seq2label, twist_seq2label = value_for_array(pred, time_steps-sliding_window+1)

        df11 = pd.DataFrame({'timestep': seq2label_times, 'pull_seq2label': pull_seq2label})
        df22 = pd.DataFrame({'timestep': seq2label_times, 'push_seq2label': push_seq2label})
        df33 = pd.DataFrame({'timestep': seq2label_times, 'shake_seq2label': shake_seq2label})
        df44 = pd.DataFrame({'timestep': seq2label_times, 'twist_seq2label': twist_seq2label})

        plt.plot(df11.timestep, df11.pull_seq2label, color="blue", label='pull', linewidth=3)
        plt.plot(df22.timestep, df22.push_seq2label, color='red', label='push', linewidth=3)
        plt.plot(df33.timestep, df33.shake_seq2label, color='green', label='shake', linewidth=3)
        plt.plot(df44.timestep, df44.twist_seq2label, color='orange', label='twist', linewidth=3)

        plt.title('sample ' + str(n) + ' /18')
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()

