#!/usr/bin/env python3

import numpy as np
from larcc_interface.config.definitions import ROOT_DIR
import json

from larcc_interface.larcc_classes.data_storage.SortedDataForLearning import (SortedDataForLearning)

if __name__ == '__main__':
    f = open(ROOT_DIR + '/data_storage/src/clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    # f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    # training_config = json.load(f)
    # f.close()

    # measurements = int(training_config["rate"] * training_config["time"])

    # # FULL TIME WINDOW ----------------------------------------------------------------------------------------
    # experiment_data = np.load(ROOT_DIR + "/data_storage/full_timewindow/data/raw_learning_data.npy")
    # array_norm = np.empty((0, experiment_data.shape[1], experiment_data.shape[2]))
    #
    # for user in range(0, experiment_data.shape[0]):
    #     data_array = experiment_data[user]
    #     result = np.reshape(data_array[:, -1], (data_array.shape[0], 1))
    #     data_array = data_array[:, :-1]
    #
    #     data_array_norm = np.empty((data_array.shape[0], 0))
    #
    #     data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
    #     data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
    #     data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
    #     data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))
    #
    #     data_array_norm = np.hstack((data_array_norm, result))
    #     array_norm = np.append(array_norm,
    #                            np.reshape(data_array_norm, (1, data_array_norm.shape[0], data_array_norm.shape[1])),
    #                            axis=0)
    #
    # np.save(ROOT_DIR + "/data_storage/full_timewindow/data/universal_normalized_data.npy", array_norm)
    # # --------------------------------------------------------------------------------------------------------

    # OLD SAMPLES ----------------------------------------------------------------------------------------
    path = ROOT_DIR + "/data_storage/data/user_splitted_raw_data/"

    time_window = 2
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    config_file = "training_config_time_" + str(time_window)

    sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    measurements = 20

    array_norm = np.empty((0, x_train.shape[1]))
    #
    for vector in x_train:

        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
        data_array_norm = np.empty((data_array.shape[0], 0))

        data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))

        vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

        array_norm = np.append(array_norm, vector_data_norm, axis=0)

    array_norm = np.append(array_norm, np.reshape([y_train], (-1, 1)), axis=1)
    np.save(ROOT_DIR + "/data_storage/data1/global_normalized_train_data_20ms.npy", array_norm)

    array_norm = np.empty((0, x_test.shape[1]))
    #
    for vector in x_test:
        data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
        data_array_norm = np.empty((data_array.shape[0], 0))

        data_array_norm = np.hstack((data_array_norm, data_array[:, 0:1] / data_max_timestamp))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 1:7] / data_max_joints))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 7:10] / data_max_gripper_F))
        data_array_norm = np.hstack((data_array_norm, data_array[:, 10:13] / data_max_gripper_M))

        vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

        array_norm = np.append(array_norm, vector_data_norm, axis=0)

    array_norm = np.append(array_norm, np.reshape([y_test], (-1, 1)), axis=1)
    np.save(ROOT_DIR + "/data_storage/data1/global_normalized_test_data_20ms.npy", array_norm)
    # --------------------------------------------------------------------------------------------------------

    # path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    # sorted_data_for_learning = SortedDataForLearning(path=path)
    #
    # training_data = sorted_data_for_learning.training_data
    # test_data = sorted_data_for_learning.test_data
    #
    # x_train = np.reshape(training_data[:, :-1], (int(training_data.shape[0] / 2), 100, 13))
    # y_train = training_data[:, -1]
    # x_test = np.reshape(test_data[:, :-1], (int(test_data.shape[0] / 2), 100, 13))
    # y_test = test_data[:, -1]
    #
    # array_norm = np.empty((0, x_train.shape[1], x_train.shape[2]))
    # array_not_norm = np.empty((0, x_train.shape[1], x_train.shape[2]))
    # for sample in x_train:
    #
    #     data_array_norm = np.empty((sample.shape[0], 0))
    #     data_array_not_norm = np.empty((sample.shape[0], 0))
    #
    #     # data_array_norm = np.hstack((data_array_norm, sample[:, 0:1] / data_max_timestamp))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 0:1]))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 1:7] / data_max_joints))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 7:10] / data_max_gripper_F))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 10:13] / data_max_gripper_M))
    #     data_array_norm = np.reshape(data_array_norm, (1, data_array_norm.shape[0], data_array_norm.shape[1]))
    #
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 0:1]))
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 1:7]))
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 7:10]))
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 10:13]))
    #     data_array_not_norm = np.reshape(data_array_not_norm,
    #                                      (1, data_array_not_norm.shape[0], data_array_not_norm.shape[1]))
    #
    #     array_norm = np.append(array_norm, data_array_norm, axis=0)
    #     array_not_norm = np.append(array_not_norm, data_array_not_norm, axis=0)
    #
    # # array_norm = np.append(array_norm, np.reshape([y_train], (-1, 1)), axis=1)
    # np.save(ROOT_DIR + "/data_storage/data2/x_train_global_normalized_data.npy", array_norm)
    # np.save(ROOT_DIR + "/data_storage/data2/x_train_raw_data.npy", array_not_norm)
    #
    # array_norm = np.empty((0, x_test.shape[1], x_test.shape[2]))
    # array_not_norm = np.empty((0, x_train.shape[1], x_train.shape[2]))
    # for sample in x_test:
    #
    #     # print("\nsample.shape\n")
    #     # print(sample.shape)
    #     #
    #     # print("\nsample\n")
    #     # print(sample[48:52, :])
    #     #
    #     # input()
    #     data_array_norm = np.empty((sample.shape[0], 0))
    #     data_array_not_norm = np.empty((sample.shape[0], 0))
    #
    #     # data_array_norm = np.hstack((data_array_norm, sample[:, 0:1] / data_max_timestamp))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 0:1]))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 1:7] / data_max_joints))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 7:10] / data_max_gripper_F))
    #     data_array_norm = np.hstack((data_array_norm, sample[:, 10:13] / data_max_gripper_M))
    #     data_array_norm = np.reshape(data_array_norm, (1, data_array_norm.shape[0], data_array_norm.shape[1]))
    #
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 0:1]))
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 1:7]))
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 7:10]))
    #     data_array_not_norm = np.hstack((data_array_not_norm, sample[:, 10:13]))
    #     data_array_not_norm = np.reshape(data_array_not_norm, (1, data_array_not_norm.shape[0], data_array_not_norm.shape[1]))
    #
    #     # print("\ndata_array_norm.shape\n")
    #     # print(data_array_norm.shape)
    #     #
    #     # print("\ndata_array_norm\n")
    #     # print(data_array_norm[:, 48:52, :])
    #
    #     array_norm = np.append(array_norm, data_array_norm, axis=0)
    #     array_not_norm = np.append(array_not_norm, data_array_not_norm, axis=0)
    #
    # # array_norm = np.append(array_norm, np.reshape([y_train], (-1, 1)), axis=1)
    # np.save(ROOT_DIR + "/data_storage/data2/x_test_global_normalized_data.npy", array_norm)
    # np.save(ROOT_DIR + "/data_storage/data2/x_test_raw_data.npy", array_not_norm)
    #
    # # TRUE RESULTS
    # y_test_final = []
    # for line in range(0, y_test.shape[0], 2):
    #     r = [int(y_test[line]), int(y_test[line + 1])]
    #     y_test_final.append(r)
    # y_test_final = np.array(y_test_final)
    # np.save(ROOT_DIR + "/data_storage/data2/y_test_data.npy", y_test_final)
    #
    # # TRUE RESULTS
    # y_train_final = []
    # for line in range(0, y_train.shape[0], 2):
    #     r = [int(y_train[line]), int(y_train[line + 1])]
    #     y_train_final.append(r)
    # y_train_final = np.array(y_train_final)
    # np.save(ROOT_DIR + "/data_storage/data2/y_train_data.npy", y_train_final)


