#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from config.definitions import ROOT_DIR


class ProcessData:
    def __init__(self, path=ROOT_DIR + "/data_storage/full_timewindow/data/", data_file="raw_learning_data.npy", div=0.7):

        self.data_norm = np.empty((0, 0))
        self.test_data_60s = np.empty((0, 0))
        f = open(ROOT_DIR + '/data_storage/full_timewindow/training_config.json')
        self.training_config = json.load(f)
        f.close()
        self.measurements = int(self.training_config["rate"] * self.training_config["time"])

        if os.path.isfile(path + data_file):
            self.experiment_data = np.load(path + data_file)

            print("self.experiment_data.shape")
            print(self.experiment_data.shape)
            self.process_data()

            self.test_data_60s = self.data_norm

            np.save("/tmp/test_data_60s.npy", self.test_data_60s)

            print("<===============================================test_data 60s=======>")
            print("Testing data 60s shape : " + str(self.test_data_60s.shape))
            print("<======================================================>")

        else:
            print("Could not find learning data file")

    def process_data(self):
        self.normalize_data(self.experiment_data, self.training_config)
        print("Normalization complete...")

    def normalize_data(self, array, train_config):
        array_norm = np.empty((array.shape[0], array.shape[1], array.shape[2]))

        for n_sample in range(0, array.shape[0]):
            sample = array[n_sample]
            learning_array = sample[:, :-1]
            data_array_norm = np.empty((learning_array.shape[0], 0))
            idx = 0
            for n in train_config["normalization_clusters"]:
                data_sub_array = learning_array[:, idx:idx+n]
                idx += n
                data_max = abs(max(data_sub_array.min(), data_sub_array.max(), key=abs))
                data_sub_array_norm = data_sub_array / data_max
                data_array_norm = np.append(data_array_norm, data_sub_array_norm, axis=1)
            true = np.reshape(sample[:, -1], (sample[:, -1].shape[0], 1))
            sample_norm = np.append(data_array_norm, true, axis=1)
            array_norm[n_sample] = sample_norm
        self.data_norm = array_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for sorter script")
    parser.add_argument("-p", "--path", type=str, default=ROOT_DIR + "/data_storage/full_timewindow/data/",
                        help="The relative path to the .npy file")
    parser.add_argument("-f", "--file", type=str, default="raw_learning_data.npy", help="The name of the .npy file")
    parser.add_argument("-d", "--div", type=float, default=0.7,
                        help="Percentage of samples that will be used for training and validation (0 to 1). The rest "
                             "will be used for tests")

    args = vars(parser.parse_args())

    processed_data = ProcessData(path=args["path"], data_file=args["file"], div=args["div"])

