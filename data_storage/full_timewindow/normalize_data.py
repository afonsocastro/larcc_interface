#!/usr/bin/env python3

import numpy as np
from larcc_interface.config.definitions import ROOT_DIR
import json

if __name__ == '__main__':
    f = open(ROOT_DIR + '/data_storage/src/clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    f = open(ROOT_DIR + '/data_storage/full_timewindow/training_config.json')
    training_config = json.load(f)
    f.close()

    measurements = int(training_config["rate"] * training_config["time"])

    experiment_data = np.load(ROOT_DIR + "/data_storage/full_timewindow/data/raw_learning_data_15s.npy")
    experiment_data = experiment_data[:,:,1:] #Remove the timestamp column
    array_norm = np.empty((0, experiment_data.shape[1], experiment_data.shape[2]))

    for sample in range(0, experiment_data.shape[0]):
        data_array = experiment_data[sample]
        result = np.reshape(data_array[:, -1], (data_array.shape[0], 1))
        data_array = data_array[:, :-1]
        # data_array = data_array[:, 1:-1]

        data_array_norm = np.empty((data_array.shape[0], 0))

        # data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 0:6] / data_max_joints))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 6:9] / data_max_gripper_F))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 9:12] / data_max_gripper_M))

        data_array_norm = np.hstack((data_array_norm, result))
        array_norm = np.append(array_norm,
                               np.reshape(data_array_norm, (1, data_array_norm.shape[0], data_array_norm.shape[1])),
                               axis=0)

    np.save(ROOT_DIR + "/data_storage/full_timewindow/data/normalized_data_15s.npy", array_norm)