#!/usr/bin/env python3
from config.definitions import ROOT_DIR
import numpy as np

if __name__ == '__main__':
    raw_data = np.load(ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/Maf_learning_data_6.npy")
    processed_data = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Maf_learning_data_6.npy")
    print("raw_data[0]")
    print(raw_data[0])
    print("processed_data[0]")
    print(processed_data[0])
