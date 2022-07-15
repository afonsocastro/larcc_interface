#!/usr/bin/env python3
import argparse
import os
import numpy as np
from sklearn.preprocessing import normalize


class SortedDataForLearning:
    def __init__(self, path="./../data/learning/", data_file="learning_data.npy", div1=0.7, div2=0.7):

        self.data_norm = np.empty((0, 0))

        self.trainning_data = np.empty((0, 0))
        self.validation_data = np.empty((0, 0))
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

            div1_idx = int(div1 * self.data_norm.shape[0])

            set1 = self.data_norm[:div1_idx]
            self.test_data = self.data_norm[div1_idx:]

            div2_idx = int(div2 * set1.shape[0])

            self.trainning_data = set1[:div2_idx]
            self.validation_data = set1[div2_idx:]

            np.save("/tmp/trainning_data.npy", self.trainning_data)
            np.save("/tmp/validation_data.npy", self.validation_data)
            np.save("/tmp/test_data.npy", self.test_data)

            print(self.trainning_data.shape)
            print(self.validation_data.shape)
            print(self.test_data.shape)

        else:
            print("Could not find learning data file")

    def get_learning_data(self):
        return self.trainning_data, self.validation_data, self.test_data

    def normalize_data(self, array, measurements):

        learning_array = array[:, :-1]
        array_norm = np.empty((0, learning_array.shape[1]))

        for vector in learning_array:

            data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
            experiment_array_norm = normalize(data_array, axis=0, norm='max')

            vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))

            array_norm = np.append(array_norm, vector_data_norm, axis=0)

        array_norm = np.append(array_norm, np.reshape([array[:, -1]], (-1, 1)), axis=1)
        print(array_norm.shape)

        self.data_norm = array_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for sorter script")
    parser.add_argument("-p", "--path", type=str, default="./../data/trainning2/",
                        help="The relative path to the .npy file")
    parser.add_argument("-f", "--file", type=str, default="learning_data.npy",
                        help="The name of the .npy file")
    parser.add_argument("-d1", "--div1", type=float, default=0.7,
                        help="Percentage of samples that will be used for trainning and validation (0 to 1). The rest "
                             "will be used for tests")
    parser.add_argument("-d2", "--div2", type=float, default=0.7,
                        help="Percentage of samples that will be used for trainning the subset created in the "
                             "first division (div1) (0 to 1). The rest will be used for validation")

    args = vars(parser.parse_args())

    sort_data_for_learing = SortedDataForLearning(path=args["path"], data_file=args["file"],
                                                  div1=args["div1"], div2=args["div2"])

