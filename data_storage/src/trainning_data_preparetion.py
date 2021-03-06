#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
from sklearn.preprocessing import normalize


class SortedDataForLearning:
    def __init__(self, path="./../data/raw_learning_data/", data_file="raw_learning_data.npy",
                 config_file="training_config", div=0.7):

        st = time.time()

        self.data_shortened = np.empty((0, 0))
        self.data_norm = np.empty((0, 0))
        self.data_filtered = np.empty((0, 0))

        self.trainning_data = np.empty((0, 0))
        self.test_data = np.empty((0, 0))

        f = open(path + '../../config/data_storage_config.json')
        storage_config = json.load(f)
        f.close()
        f = open(path + '../../config/' + config_file + '.json')
        training_config = json.load(f)
        f.close()

        measurements = int(storage_config["rate"] * storage_config["time"])

        files = os.listdir(path)
        file_exist = False

        for file in files:
            if file.find(data_file) != -1:
                file_exist = True

        if file_exist:
            experiment_data = np.load(path + data_file)

            self.sample_shortener(experiment_data, measurements, storage_config, training_config)

            self.filter_data(self.data_shortened, storage_config, training_config)

            self.normalize_data(self.data_filtered, storage_config, training_config)

            np.random.shuffle(self.data_norm)

            div_idx = int(div * self.data_norm.shape[0])

            self.trainning_data = self.data_norm[:div_idx]
            self.test_data = self.data_norm[div_idx:]

            np.save("/tmp/trainning_data.npy", self.trainning_data)
            np.save("/tmp/test_data.npy", self.test_data)

            print("Learning data shape")
            print(self.trainning_data.shape)
            print("Testing data shape")
            print(self.test_data.shape)

        else:
            print("Could not find learning data file")

        print("Script run for: " + str(round((time.time() - st), 2)) + " seconds")

    def sample_shortener(self, array, measurements, store_config, train_config):

        new_measurements = store_config["rate"] * train_config["time"]
        array_shortened = np.empty((0, int(new_measurements * len(store_config["data"]))))

        classification = []
        for vector in array:
            data_array = np.reshape(vector[:-1], (measurements, int(len(vector) / measurements)))
            idx_start = 0
            idx_end = int(new_measurements)

            c = vector[-1]

            while True:
                if idx_end > measurements:
                    break

                vector_shortened = np.reshape(data_array[idx_start:idx_end, :], (1, array_shortened.shape[1]))
                array_shortened = np.append(array_shortened, vector_shortened, axis=0)
                classification.append(c)
                idx_start += int(new_measurements)
                idx_end += int(new_measurements)

        array_shortened = np.append(array_shortened, np.reshape([classification], (-1, 1)), axis=1)
        self.data_shortened = array_shortened

    def normalize_data(self, array, store_config, train_config):

        measurements = int(store_config["rate"] * train_config["time"])

        learning_array = array[:, :-1]
        array_norm = np.empty((0, learning_array.shape[1]))

        for vector in learning_array:

            data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
            experiment_array_norm = normalize(data_array, axis=0, norm='max')

            vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))

            array_norm = np.append(array_norm, vector_data_norm, axis=0)

        array_norm = np.append(array_norm, np.reshape([array[:, -1]], (-1, 1)), axis=1)

        self.data_norm = array_norm

    def filter_data(self, array, store_config, train_config):

        measurements = int(store_config["rate"] * train_config["time"])
        data = store_config["data"]
        data_filtered = train_config["data_filtered"]

        list_idx = []
        for filtered in data_filtered:
            list_idx.append(data.index(filtered))

        learning_array = array[:, :-1]
        array_filtered = np.empty((0, int(len(list_idx) * measurements)))

        for vector in learning_array:
            data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
            data_array_filtered = np.empty((data_array.shape[0], 0))

            for idx in list_idx:
                data_array_filtered = np.append(data_array_filtered, np.reshape([data_array[:, idx]], (-1, 1)), axis=1)

            vector_data_filtered = np.reshape(data_array_filtered, (1, int(len(list_idx) * measurements)))

            array_filtered = np.append(array_filtered, vector_data_filtered, axis=0)

        array_filtered = np.append(array_filtered, np.reshape([array[:, -1]], (-1, 1)), axis=1)

        self.data_filtered = array_filtered


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for sorter script")
    parser.add_argument("-p", "--path", type=str, default="./../data/raw_learning_data/",
                        help="The relative path to the .npy file")
    parser.add_argument("-f", "--file", type=str, default="raw_learning_data.npy",
                        help="The name of the .npy file")
    parser.add_argument("-d", "--div", type=float, default=0.7,
                        help="Percentage of samples that will be used for trainning and validation (0 to 1). The rest "
                             "will be used for tests")
    parser.add_argument("-c", "--config_file", type=str, default="training_config",
                        help="If argmument is present, activates gripper")

    args = vars(parser.parse_args())

    sort_data_for_learing = SortedDataForLearning(path=args["path"], data_file=args["file"],
                                                  config_file=args["config_file"], div=args["div"])

