#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import normalize


def normalize_data(vector, measurements):

    data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
    experiment_array_norm = normalize(data_array, axis=0, norm='max')

    vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))
    return vector_data_norm


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


A = np.array([[20, 0, -10],
             [0, 1, 0],
             [-3, 1, 4]])
print("Original matrix:")
print(A)
print("-------------------------------\nNormalized matrix (by columns):")
B = normalize(A, axis=0, norm="max")
print(B)
