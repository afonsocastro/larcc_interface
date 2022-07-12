#!/usr/bin/env python3
import argparse
import os
import numpy as np


class SortedDataForLearning:
    def __init__(self, path="./../data/trainning2/", data_file="learning_data.npy", div1=0.7, div2=0.7):

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

            np.random.shuffle(experiment_data)

            div1_idx = int(div1 * experiment_data.shape[0])

            set1 = experiment_data[:div1_idx]
            self.test_data = experiment_data[div1_idx:]

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
