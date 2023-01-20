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
    validation_split = 0.3
    batch_size = 2
    # time_steps = 50
    time_steps = 100
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50

    # load array
    pred_rnn = load('Bahdanau_Attention_model_pred.npy')
    pred_cnn = load('cnn4_model_pred.npy')
    true = load('true_results_cnn4_vs_rnnAttention.npy')

    times = np.array([i for i in range(100)])
    cnn_times = np.array([i for i in range(19, 100)])

    # for n in range(0, len(pred_rnn)):
    for n in range(0, len(pred_cnn)):

        fig = plt.figure(figsize=(12, 8))

        # RECURRENT GRAPH
        pull_rnn, push_rnn, shake_rnn, twist_rnn = value_for_array(pred_rnn, n, 100)
        df1 = pd.DataFrame({'timestep': times, 'pull_rnn': pull_rnn})
        df2 = pd.DataFrame({'timestep': times, 'push_rnn': push_rnn})
        df3 = pd.DataFrame({'timestep': times, 'shake_rnn': shake_rnn})
        df4 = pd.DataFrame({'timestep': times, 'twist_rnn': twist_rnn})

        plt.plot(df1.timestep, df1.pull_rnn, color='darkblue', label='pull R', linewidth=3)
        plt.plot(df2.timestep, df2.push_rnn, color='darkgreen', label='push R', linewidth=3)
        plt.plot(df3.timestep, df3.shake_rnn, color='darkorange', label='shake R', linewidth=3)
        plt.plot(df4.timestep, df4.twist_rnn, color='maroon', label='twist R', linewidth=3)

        # CONVOLUTIONAL GRAPH
        pull_cnn, push_cnn, shake_cnn, twist_cnn = value_for_array(pred_cnn, n, 81)

        df11 = pd.DataFrame({'timestep': cnn_times, 'pull_cnn': pull_cnn})
        df22 = pd.DataFrame({'timestep': cnn_times, 'push_cnn': push_cnn})
        df33 = pd.DataFrame({'timestep': cnn_times, 'shake_cnn': shake_cnn})
        df44 = pd.DataFrame({'timestep': cnn_times, 'twist_cnn': twist_cnn})

        plt.plot(df11.timestep, df11.pull_cnn, color="deepskyblue", label='pull C', linewidth=3)
        plt.plot(df22.timestep, df22.push_cnn, color='limegreen', label='push C', linewidth=3)
        plt.plot(df33.timestep, df33.shake_cnn, color='gold', label='shake C', linewidth=3)
        plt.plot(df44.timestep, df44.twist_cnn, color='red', label='twist C', linewidth=3)

        results_true = string_result(true[n])
        plt.title('Primitives confidences at timestep.   True Expected: ' + results_true[0] + ' and ' + results_true[1])
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()

