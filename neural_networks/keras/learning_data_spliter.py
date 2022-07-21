#!/usr/bin/env python3

import numpy as np

training_percent = 0.7

all_data = np.load('../data/learning_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                   encoding='ASCII')

n_samples = all_data.shape[0]
n_split = int(training_percent * n_samples)

np.random.shuffle(all_data)

training_data = all_data[:n_split]
test_data = all_data[n_split:]

print(training_data.shape)
print(test_data.shape)

np.save('../data/learning_data_training.npy', training_data)
np.save('../data/learning_data_test.npy', test_data)
