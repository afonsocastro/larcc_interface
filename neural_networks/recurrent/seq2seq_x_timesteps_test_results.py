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
    pred = load('pred_model_Bahdanau_Attention.npy')
    true = load('true_model_Bahdanau_Attention.npy')

    # print the array
    # print(data)

    times = np.array([i for i in range(100)])

    for n in range(0,len(pred)):
        pull, push, shake, twist = value_for_array(pred, n, 100)

        print(true[n])

        fig = plt.figure(figsize=(12, 8))

        df1 = pd.DataFrame({'timestep': times, 'pull': pull})
        df2 = pd.DataFrame({'timestep': times, 'push': push})
        df3 = pd.DataFrame({'timestep': times, 'shake': shake})
        df4 = pd.DataFrame({'timestep': times, 'twist': twist})

        plt.plot(df1.timestep, df1.pull, label='pull', linewidth=3)
        plt.plot(df2.timestep, df2.push, color='red', label='push', linewidth=3)
        plt.plot(df3.timestep, df3.shake, color='green', label='shake', linewidth=3)
        plt.plot(df4.timestep, df4.twist, color='orange', label='twist', linewidth=3)

        plt.title('Primitives confidences at timestep ')
        plt.xlabel('timestep')
        plt.ylabel('Confidence')
        plt.legend()
        plt.tight_layout()
        plt.show()


    exit(0)

    pull_0, push_0, shake_0, twist_0 = value_for_array(pred, 0, 100)
    pull_1, push_1, shake_1, twist_1 = value_for_array(pred, 1, 100)

    # Create figure

    fig = plt.figure(figsize=(12, 8))

    # Define Data

    df1 = pd.DataFrame({'timestep': np.array([i for i in range(100)]), 'pull': pull_0})
    df2 = pd.DataFrame({'timestep': np.array([i for i in range(100)]), 'push': push_0})
    df3 = pd.DataFrame({'timestep': np.array([i for i in range(100)]), 'shake': shake_0})
    df4 = pd.DataFrame({'timestep': np.array([i for i in range(100)]), 'twist': twist_0})

    # Plot time series

    plt.plot(df1.timestep, df1.pull, label='pull', linewidth=3)
    plt.plot(df2.timestep, df2.push, color='red', label='push', linewidth=3)
    plt.plot(df3.timestep, df3.shake, color='green', label='shake', linewidth=3)
    plt.plot(df4.timestep, df4.twist, color='orange', label='twist', linewidth=3)

    # Add title and labels

    plt.title('Primitives confidences at timestep ')
    plt.xlabel('timestep')
    plt.ylabel('Confidence')

    # Add legend

    plt.legend()

    # Auto space

    plt.tight_layout()

    # Display plot

    plt.show()
