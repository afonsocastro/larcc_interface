#!/usr/bin/env python3

import numpy as np
from config.definitions import ROOT_DIR
import json


if __name__ == '__main__':
    f = open(ROOT_DIR + '/data_storage/src/clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    training_config = json.load(f)
    f.close()

    measurements = int(training_config["rate"] * training_config["time"])

    experiment_data = np.load(ROOT_DIR + "/data_storage/data/raw_learning_data/raw_learning_data.npy")

    learning_array = experiment_data[:, :-1]

    array_norm = np.empty((0, learning_array.shape[1]))

    for vector in learning_array:

        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
        data_array_norm = np.empty((data_array.shape[0], 0))

        data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))

        vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

        array_norm = np.append(array_norm, vector_data_norm, axis=0)

    array_norm = np.append(array_norm, np.reshape([experiment_data[:, -1]], (-1, 1)), axis=1)
    np.save(ROOT_DIR + "/data_storage/data/universal_norm/normalized_data.npy", array_norm)

