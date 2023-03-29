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
                color = "red"
            elif t[i-1] == 1:
                color = "green"
            elif t[i-1] == 2:
                color = "blue"
            elif t[i-1] == 3:
                color = "orange"
            p.axvspan(start, end, color=color, alpha=0.2, lw=0)
            start = end


if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 2
    # time_steps = 50
    time_steps = 6000
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    # load array
    pred_rnn = load('25_results/predicted.npy')
    all_true = load('25_results/y_true.npy')
    # pred_rnn = load('predicted.npy')
    # true = load('y_true.npy')

    times = np.array([i for i in range(6000)])
    print(pred_rnn.shape)
    print(all_true.shape)

    for n in range(0, pred_rnn.shape[0]):
        pred = pred_rnn[n]
        true = all_true[n]


        # fig, ax = plt.subplots()

        # pull_rnn, push_rnn, shake_rnn, twist_rnn = value_for_array(pred_rnn, n, 6000)
        pull_rnn, push_rnn, shake_rnn, twist_rnn = value_for_array(pred, 6000)
        df1 = pd.DataFrame({'timestep': times, 'pull_rnn': pull_rnn})
        df2 = pd.DataFrame({'timestep': times, 'push_rnn': push_rnn})
        df3 = pd.DataFrame({'timestep': times, 'shake_rnn': shake_rnn})
        df4 = pd.DataFrame({'timestep': times, 'twist_rnn': twist_rnn})

        plt.plot(df1.timestep, df1.pull_rnn, color='red', label='pull R', linewidth=3)
        plt.plot(df2.timestep, df2.push_rnn, color='green', label='push R', linewidth=3)
        plt.plot(df3.timestep, df3.shake_rnn, color='blue', label='shake R', linewidth=3)
        plt.plot(df4.timestep, df4.twist_rnn, color='orange', label='twist R', linewidth=3)

        plot_true_shadow(times, true, plt)

        plt.title('Primitives confidences at timestep. 50-timesteps RNN / 25 steps sliding window  ')
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()
