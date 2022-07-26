#!/usr/bin/env python3
import json
import math
import time

import numpy as np
import rospy
from sklearn.preprocessing import normalize
from data_aquisition_node import DataForLearning
from colorama import Fore


# def normalize_data(vector, measurements):
#
#     data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
#     experiment_array_norm = normalize(data_array, axis=0, norm='max')
#
#     vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))
#     return vector_data_norm
#
#
# def sample_shortener(array, measurements, store_config, train_config):
#
#     new_measurements = store_config["rate"] * train_config["time"]
#     array_shortened = np.empty((0, int(new_measurements * len(store_config["data"]))))
#
#     classification = []
#     for vector in array:
#         data_array = np.reshape(vector[:-1], (measurements, int(len(vector) / measurements)))
#         idx_start = 0
#         idx_end = int(new_measurements)
#
#         c = vector[-1]
#
#         while True:
#             if idx_end > measurements:
#                 break
#
#             vector_shortened = np.reshape(data_array[idx_start:idx_end, :], (1, array_shortened.shape[1]))
#             array_shortened = np.append(array_shortened, vector_shortened, axis=0)
#             classification.append(c)
#             idx_start += int(new_measurements)
#             idx_end += int(new_measurements)
#
#     array_shortened = np.append(array_shortened, np.reshape([classification], (-1, 1)), axis=1)


rospy.init_node("test_aquisition", anonymous=True)

rate = rospy.Rate(50)

data_for_learning = DataForLearning()

time.sleep(0.2)

i = 0

data = {"fx": [],
        "fy": [],
        "fz": [],
        "mx": [],
        "my": [],
        "mz": []}

while not rospy.is_shutdown():
    print(data_for_learning)

    current_data = data_for_learning

    data["fx"].append(current_data.wrench_force_torque.force.x)
    data["fy"].append(current_data.wrench_force_torque.force.y)
    data["fz"].append(current_data.wrench_force_torque.force.z)
    data["mx"].append(current_data.wrench_force_torque.torque.x)
    data["my"].append(current_data.wrench_force_torque.torque.y)
    data["mz"].append(current_data.wrench_force_torque.torque.z)

    if i > 100:
        break

    i += 1
    rate.sleep()

for key in data:
    val_min = str(min(data[key]))
    val_max = str(max(data[key]))
    val_len = len(data[key])
    val_sum = sum(data[key])
    val_average = float(val_sum / val_len)

    deviations = [(x - val_average) ** 2 for x in data[key]]

    variance = sum(deviations) / val_len

    SD = math.sqrt(variance)

    CI = 3.29 * (SD / math.sqrt(val_len))

    print(Fore.BLUE + key + ":" + Fore.RESET + "\nMean: " + str(val_average) + '\nVariance: ' + str(SD) +
          "\nConfidance Interval: " + str(CI))


# path = "./../data/trainning2/"
# config_file = "training_config"
#
# f = open(path + '../../config/data_storage_config.json')
# storage_config = json.load(f)
# f.close()
# f = open(path + '../../config/' + config_file + '.json')
# training_config = json.load(f)
# f.close()
# data_file = "learning_data.npy"
#
# experiment_data = np.load(path + data_file)
#
# sample_shortener(experiment_data, 50, storage_config, training_config)



# path = "../data/trainning/"
# file = "../data/trainning2/test_array.npy"
#
# array = np.empty((0, 10))
# for i in range(0, 50):
#     row = np.ones((1, 10)) * i
#     array = np.append(array, row, axis=0)
#
# # np.save(path + file, array)
#
# print(array)
#
# np.random.shuffle(array)
#
# print(array)

# print("Original matrix:")
# print(A)
# print("-------------------------------\nNormalized matrix (by columns):")
# B = normalize(A, axis=0, norm="max")
# print(B)

# dt = np.dtype('float32', metadata={"key": "value"})
#
# print(dt.metadata["key"])
#
# arr = np.array([1, 2, 3], dtype=dt)
#
# print(arr.dtype.metadata)
# print(arr.dtype)
# A = np.load("../data/raw_learning_data/raw_learning_data.npy", allow_pickle=True)
# print(A)
# print(type(A))
# print(A.items())
