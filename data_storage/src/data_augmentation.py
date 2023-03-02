#!/usr/bin/env python3

import numpy as np
from config.definitions import ROOT_DIR
import json

if __name__ == '__main__':
    f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    training_config = json.load(f)
    f.close()

    measurements = int(training_config["rate"] * training_config["time"])

    experiment_data = np.load(ROOT_DIR + "/data_storage/data/universal_norm/normalized_data.npy")

    learning_array = experiment_data[:, 1:-1]

    learning_array_9 = learning_array * 0.9
    learning_array_8 = learning_array * 0.8
    learning_array_7 = learning_array * 0.7
    learning_array_6 = learning_array * 0.6
    learning_array_5 = learning_array * 0.5

    learning_array_9 = np.hstack((experiment_data[:, 0:1], learning_array_9, experiment_data[:, 650:651]))
    learning_array_8 = np.hstack((experiment_data[:, 0:1], learning_array_8, experiment_data[:, 650:651]))
    learning_array_7 = np.hstack((experiment_data[:, 0:1], learning_array_7, experiment_data[:, 650:651]))
    learning_array_6 = np.hstack((experiment_data[:, 0:1], learning_array_6, experiment_data[:, 650:651]))
    learning_array_5 = np.hstack((experiment_data[:, 0:1], learning_array_5, experiment_data[:, 650:651]))

    np.save(ROOT_DIR + "/data_storage/data/universal_norm/augmented_data/normalized_data_09.npy", learning_array_9)
    np.save(ROOT_DIR + "/data_storage/data/universal_norm/augmented_data/normalized_data_08.npy", learning_array_8)
    np.save(ROOT_DIR + "/data_storage/data/universal_norm/augmented_data/normalized_data_07.npy", learning_array_7)
    np.save(ROOT_DIR + "/data_storage/data/universal_norm/augmented_data/normalized_data_06.npy", learning_array_6)
    np.save(ROOT_DIR + "/data_storage/data/universal_norm/augmented_data/normalized_data_05.npy", learning_array_5)



