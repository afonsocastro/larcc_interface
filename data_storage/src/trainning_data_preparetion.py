#!/usr/bin/env python3
import argparse
import os
import numpy as np
from sklearn.preprocessing import normalize


class SortedDataForLearning:
    def __init__(self, path="./../data/raw_learning_data/", data_file="raw_learning_data.npy", div=0.7):

        self.data_norm = np.empty((0, 0))

        self.trainning_data = np.empty((0, 0))
        self.test_data = np.empty((0, 0))

        files = os.listdir(path)
        file_exist = False

        print(path + data_file)
        print(files)
        for file in files:
            if file.find(data_file) != -1:
                file_exist = True

        if file_exist:
            experiment_data = np.load(path + data_file)

            self.normalize_data(experiment_data, 50)

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

    def get_learning_data(self):
        return self.trainning_data, self.test_data

    def normalize_data(self, array, measurements):

        learning_array = array[:, :-1]
        array_norm = np.empty((0, learning_array.shape[1]))

        for vector in learning_array:

            data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
            experiment_array_norm = normalize(data_array, axis=0, norm='max')

            vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))

            array_norm = np.append(array_norm, vector_data_norm, axis=0)

        array_norm = np.append(array_norm, np.reshape([array[:, -1]], (-1, 1)), axis=1)

        self.data_norm = array_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for sorter script")
    parser.add_argument("-p", "--path", type=str, default="./../data/raw_learning_data/",
                        help="The relative path to the .npy file")
    parser.add_argument("-f", "--file", type=str, default="raw_learning_data.npy",
                        help="The name of the .npy file")
    parser.add_argument("-d", "--div", type=float, default=0.7,
                        help="Percentage of samples that will be used for trainning and validation (0 to 1). The rest "
                             "will be used for tests")

    args = vars(parser.parse_args())

    sort_data_for_learing = SortedDataForLearning(path=args["path"], data_file=args["file"], div=args["div"])

