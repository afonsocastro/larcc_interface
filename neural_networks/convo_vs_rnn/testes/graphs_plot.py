#!/usr/bin/env python3

from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.transforms as mtransforms


# def value_for_array(data, n, timesteps):
#     pull = np.array([data[n][j][0] for j in range(timesteps)])
#     push = np.array([data[n][j][1] for j in range(timesteps)])
#     shake = np.array([data[n][j][2] for j in range(timesteps)])
#     twist = np.array([data[n][j][3] for j in range(timesteps)])
#
#     return pull, push, shake, twist


def value_for_array(data, timesteps):
    pull = np.array([data[j][0] for j in range(timesteps)])
    push = np.array([data[j][1] for j in range(timesteps)])
    shake = np.array([data[j][2] for j in range(timesteps)])
    twist = np.array([data[j][3] for j in range(timesteps)])

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

def color_result(true):
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
    pred_rnn = load('predicted.npy')
    true = load('y_true.npy')

    times = np.array([i for i in range(6000)])
    print(pred_rnn.shape)
    print(true.shape)

    # for n in range(0, len(pred_rnn)):
    # use data coordinates for the x-axis and the axes coordinates for the y-axis

    # fig, ax = plt.subplots()

    # pull_rnn, push_rnn, shake_rnn, twist_rnn = value_for_array(pred_rnn, n, 6000)
    pull_rnn, push_rnn, shake_rnn, twist_rnn = value_for_array(pred_rnn, 6000)
    df1 = pd.DataFrame({'timestep': times, 'pull_rnn': pull_rnn})
    df2 = pd.DataFrame({'timestep': times, 'push_rnn': push_rnn})
    df3 = pd.DataFrame({'timestep': times, 'shake_rnn': shake_rnn})
    df4 = pd.DataFrame({'timestep': times, 'twist_rnn': twist_rnn})

    plt.plot(df1.timestep, df1.pull_rnn, color='red', label='pull R', linewidth=3)
    plt.plot(df2.timestep, df2.push_rnn, color='green', label='push R', linewidth=3)
    plt.plot(df3.timestep, df3.shake_rnn, color='blue', label='shake R', linewidth=3)
    plt.plot(df4.timestep, df4.twist_rnn, color='orange', label='twist R', linewidth=3)
    start = 0
    for i in times:
        if (i != 0 and true[i] != true[i-1]) or i == 5999:
            end = i
            if true[i-1] == 0:
                color = "red"
            elif true[i-1] == 1:
                color = "green"
            elif true[i-1] == 2:
                color = "blue"
            elif true[i-1] == 3:
                color = "orange"
            plt.axvspan(start, end, color=color, alpha=0.2, lw=0)
            start = end

    plt.title('Primitives confidences at timestep')
    plt.xlabel('timestep')
    plt.ylabel('Confidence')
    plt.legend()
    plt.tight_layout()
    plt.show()
