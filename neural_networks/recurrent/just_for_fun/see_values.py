#!/usr/bin/env python3
import numpy as np

from config.definitions import ROOT_DIR

if __name__ == '__main__':
    test_data = np.load(ROOT_DIR + "/data_storage/data3/global_normalized_data.npy")

    print("test_data[0, :, 0]")
    print(test_data[0, :, 0])
