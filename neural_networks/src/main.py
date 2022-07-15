#!/usr/bin/env python3

import network
import numpy as np


if __name__ == '__main__':

    # tr_data = np.load('../data/trainning_data_norm.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    tr_data = np.load('../../data_storage/data/trainning/trainning_data_norm.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    # tr_data_results = np.load('../data/trainning_data_results.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    tr_data_results = np.load('../../data_storage/data/trainning/trainning_data_results.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    # te_data = np.load('../data/test_data_norm.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    te_data = np.load('../../data_storage/data/trainning/test_data_norm.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    # te_data_results = np.load('../data/test_data_results.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    te_data_results = np.load('../../data_storage/data/trainning/test_data_results.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    row, col = tr_data.shape

    training_inputs = [x.reshape(-1, 1) for x in tr_data]
    training_results = [y.reshape(-1, 1) for y in tr_data_results]
    training_data = list(zip(training_inputs, training_results))
    test_inputs = [x.reshape(-1, 1) for x in te_data]
    test_data = list(zip(test_inputs, te_data_results))

    net = network.Network([col, 30, 3])
    # net.SGD(training_data, 30, 10, 3.0)

    # def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    net.SGD(training_data, 100, 10, 2.0, test_data=test_data)



