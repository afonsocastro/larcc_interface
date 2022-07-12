#!/usr/bin/env python3
import numpy as np


path = "../data/trainning/"
file = "../data/trainning2/test_array.npy"

array = np.empty((0, 10))
for i in range(0, 50):
    row = np.ones((1, 10)) * i
    array = np.append(array, row, axis=0)

# np.save(path + file, array)

print(array)

np.random.shuffle(array)

print(array)
