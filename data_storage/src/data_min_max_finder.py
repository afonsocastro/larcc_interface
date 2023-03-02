#!/usr/bin/env python3

import numpy as np
from config.definitions import ROOT_DIR
import json


if __name__ == '__main__':

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    training_config = json.load(f)
    f.close()

    measurements = int(training_config["rate"] * training_config["time"])

    experiment_data = np.load(ROOT_DIR + "/data_storage/data/raw_learning_data/raw_learning_data.npy")

    learning_array = experiment_data[:, :-1]

    clusters_max = {"timestamp": {"max": 0, "min": 0}, "joints": {"max": 0, "min": 0},
                    "gripper_F": {"max": 0, "min": 0}, "gripper_M": {"max": 0, "min": 0}}

    for vector in learning_array:
        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))

        idx = 0
        cont = 0
        for n in training_config["normalization_clusters"]:
            data_sub_array = data_array[:, idx:idx + n]
            idx += n

            min = data_sub_array.min()
            max = data_sub_array.max()

            if cont == 0:
                if min < clusters_max["timestamp"]["min"]:
                    clusters_max["timestamp"]["min"] = min
                elif max > clusters_max["timestamp"]["max"]:
                    clusters_max["timestamp"]["max"] = max

            elif cont == 1:
                if min < clusters_max["joints"]["min"]:
                    clusters_max["joints"]["min"] = min
                elif max > clusters_max["joints"]["max"]:
                    clusters_max["joints"]["max"] = max
            elif cont == 2:
                if min < clusters_max["gripper_F"]["min"]:
                    clusters_max["gripper_F"]["min"] = min
                elif max > clusters_max["gripper_F"]["max"]:
                    clusters_max["gripper_F"]["max"] = max
            elif cont == 3:
                if min < clusters_max["gripper_M"]["min"]:
                    clusters_max["gripper_M"]["min"] = min
                elif max > clusters_max["gripper_M"]["max"]:
                    clusters_max["gripper_M"]["max"] = max
            else:
                print("WHAAAAAATTTTTT?????????")

            cont += 1

    print(clusters_max)
    with open("clusters_max_min.json", "w") as fp:
        json.dump(clusters_max, fp)
