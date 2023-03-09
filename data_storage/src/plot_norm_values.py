#!/usr/bin/env python3

import numpy as np
from config.definitions import ROOT_DIR
import json
from matplotlib import pyplot as plt


if __name__ == '__main__':

    f = open(ROOT_DIR + '/data_storage/src/clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    training_config = json.load(f)
    f.close()

    measurements = int(training_config["rate"] * training_config["time"])

    experiment_data = np.load(ROOT_DIR + "/data_storage/data/raw_learning_data/raw_learning_data.npy")

    learning_array = experiment_data[:, :-1]

    clusters_list = []
    for vector in learning_array:
        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
        clusters_max = {"timestamp": 0, "joints": 0, "gripper_F": 0, "gripper_M": 0}

        idx = 0
        cont = 0
        for n in training_config["normalization_clusters"]:
            data_sub_array = data_array[:, idx:idx + n]
            idx += n

            data_max = abs(max(data_sub_array.min(), data_sub_array.max(), key=abs))
            if cont == 0:
                clusters_max["timestamp"] = data_max
            elif cont == 1:
                clusters_max["joints"] = data_max
            elif cont == 2:
                clusters_max["gripper_F"] = data_max
            elif cont == 3:
                clusters_max["gripper_M"] = data_max
            else:
                print("WHAAAAAATTTTTT?????????")
            cont += 1

        clusters_list.append(clusters_max)

    x = np.arange(0, len(clusters_list))

    t = abs(max(clusters_max_min["timestamp"]["min"], clusters_max_min["timestamp"]["max"], key=abs))
    j = abs(max(clusters_max_min["joints"]["min"], clusters_max_min["joints"]["max"], key=abs))
    gf = abs(max(clusters_max_min["gripper_F"]["min"], clusters_max_min["gripper_F"]["max"], key=abs))
    gm = abs(max(clusters_max_min["gripper_M"]["min"], clusters_max_min["gripper_M"]["max"], key=abs))

    timestamp = [t["timestamp"] for t in clusters_list]
    joints = [t["joints"] for t in clusters_list]
    gripper_F = [t["gripper_F"] for t in clusters_list]
    gripper_M = [t["gripper_M"] for t in clusters_list]

    fig1, axs1 = plt.subplots(2)
    fig1.suptitle('Normalization values for each sample')
    axs1[0].plot(x, timestamp)
    axs1[0].plot(x, [t] * len(x), 'tab:brown')
    axs1[0].set_title("timestamp")
    axs1[1].plot(x, joints, 'tab:orange')
    axs1[1].plot(x, [j] * len(x), 'tab:brown')
    axs1[1].set_title("joints")

    fig2, axs2 = plt.subplots(2)
    fig2.suptitle('Normalization values for each sample')
    axs2[0].plot(x, gripper_F, 'tab:green')
    axs2[0].plot(x, [gf] * len(x), 'tab:brown')
    axs2[0].set_title("gripper_F")
    axs2[1].plot(x, gripper_M, 'tab:red')
    axs2[1].plot(x, [gm] * len(x), 'tab:brown')
    axs2[1].set_title("gripper_M")

    plt.show()
