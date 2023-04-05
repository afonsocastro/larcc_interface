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

    pred_tst = load('ts_transformer_data2_pred.npy')
    true = load('true_results_data2.npy')

    tst_times = np.array([i for i in range(19, 100)])

    for n in range(0, len(pred_tst)):

        fig = plt.figure(figsize=(12, 8))

        # TS Transformer GRAPH
        pull_tst, push_tst, shake_tst, twist_tst = value_for_array(pred_tst, n, 81)

        df11 = pd.DataFrame({'timestep': tst_times, 'pull_tst': pull_tst})
        df22 = pd.DataFrame({'timestep': tst_times, 'push_tst': push_tst})
        df33 = pd.DataFrame({'timestep': tst_times, 'shake_tst': shake_tst})
        df44 = pd.DataFrame({'timestep': tst_times, 'twist_tst': twist_tst})

        plt.plot(df11.timestep, df11.pull_tst, color="blue", label='pull', linewidth=3)
        plt.plot(df22.timestep, df22.push_tst, color='red', label='push', linewidth=3)
        plt.plot(df33.timestep, df33.shake_tst, color='green', label='shake', linewidth=3)
        plt.plot(df44.timestep, df44.twist_tst, color='orange', label='twist', linewidth=3)

        # Add the vertical line to the plot
        plt.axvline(x=50, linestyle="--")

        results_true = string_result(true[n])
        # plt.title('CNN output confidences at timestep. True Expected: ' + results_true[0] + ' and ' + results_true[1])
        plt.title("Ground Truth:    " + results_true[0] + '  -->  ' + results_true[1], fontdict={"fontsize": 16, "fontweight": "bold"})
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()

