#!/usr/bin/env python3
import json

from larcc_interface.config.definitions import ROOT_DIR
import numpy as np


if __name__ == '__main__':

    # data = np.load(ROOT_DIR + '/data_storage/data/processed_learning_data/Ine_learning_data_8_old_processed.npy',
    #                     mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    data = np.load(ROOT_DIR + '/data_storage/data/new_acquisition/user_splitted_data/Joe_learning_data_11.npy',
                   mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    # print("type(data)")
    # print(type(data))
    # print("data.shape")
    # print(data.shape)

    data = np.delete(data, [13, 37, 39, 49, 52, 67, 82, 90, 110], 0)

    # np.save(ROOT_DIR + '/data_storage/data/processed_learning_data/Ine_learning_data_8.npy', data)
    np.save(ROOT_DIR + '/data_storage/data/user_splitted_raw_data/Joe_learning_data_11.npy', data)
