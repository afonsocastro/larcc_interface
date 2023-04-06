#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
from sklearn.preprocessing import normalize
from config.definitions import ROOT_DIR


class SortedDataForLearning:
    def __init__(self, path=ROOT_DIR + "/data_storage/data/raw_learning_data/", data_file="raw_learning_data.npy",
                 config_file="training_config", div=0.7):

        st = time.time()

        self.data_shortened = np.empty((0, 0))
        self.data_norm = np.empty((0, 0))
        self.data_filtered = np.empty((0, 0))
        self.data_classes_filtered = np.empty((0, 0))

        self.training_data = np.empty((0, 0))
        self.test_data = np.empty((0, 0))

        f = open(ROOT_DIR + '/data_storage/config/data_storage_config.json')
        self.storage_config = json.load(f)
        f.close()
        f = open(ROOT_DIR + '/data_storage/config/' + config_file + '.json')
        self.training_config = json.load(f)
        f.close()

        self.measurements = int(self.storage_config["rate"] * self.storage_config["time"])

        # files = os.listdir(path)
        # file_exist = False

        # for file in files:
        #     if file.find(data_file) != -1:
        #         file_exist = True
        # print("os.path.isdir(path)")
        # print(os.path.isdir(path))
        # print("path")
        # print(path)
        # print("data_file")
        # print(data_file)
        # print("config_file")
        # print(config_file)
        # print("os.path.isfile(path + data_file)")
        # print(os.path.isfile(path + data_file))
        # print("os.path.isdir(path)")
        # print(os.path.isdir(path))

        if os.path.isfile(path + data_file):
            self.experiment_data = np.load(path + data_file)

            self.process_data()
            # self.filter_classes(experiment_data, storage_config, training_config)
            # print("Class filter complete...")
            #
            # if storage_config["time"] == training_config["time"]:
            #     self.data_shortened = self.data_classes_filtered
            # else:
            #     self.sample_shortener(self.data_classes_filtered, measurements, storage_config, training_config)
            #     # self.sample_shortener(experiment_data, measurements, storage_config, training_config)
            #     print("Time truncation complete...")
            #
            # # if len(storage_config["data"]) == len(training_config["data_filtered"]):
            # if len(storage_config["data"]) == len(training_config["data"]):
            #     self.data_filtered = self.data_shortened
            # else:
            #     self.filter_data(self.data_shortened, storage_config, training_config)
            #     print("Variable filter complete...")
            #
            # self.normalize_data(self.data_filtered, storage_config, training_config)
            # print("Normalization complete...")
            #
            # np.random.shuffle(self.data_norm)
            # print("Shuffle complete...")

            div_idx = int(div * self.data_norm.shape[0])

            self.training_data = self.data_norm[:div_idx]
            self.test_data = self.data_norm[div_idx:]

            # np.save("/tmp/training_data.npy", self.training_data)
            # np.save("/tmp/test_data.npy", self.test_data)
            #
            # print("<===============================================test_data=======>")
            # print("Learning data shape: " + str(self.training_data.shape))
            # print("Testing data shape: " + str(self.test_data.shape))
            # print("<======================================================>")
        elif os.path.isdir(path):
            self.raw_training_data = np.empty((0, 651))
            self.raw_test_data = np.empty((0, 651))

            files = os.listdir(path)
            for file in files:
                new_array = np.load(path + file)
                for user in self.training_config["training_users"]:
                    number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
                    if user == number:
                        print("training data: " + str(user) )
                        print("file: " + file)
                        self.raw_training_data = np.append(self.raw_training_data, new_array, axis=0)

                for user in self.training_config["test_users"]:
                    number = int(''.join([str(x) for x in [int(s) for s in str(file) if s.isdigit()]]))
                    if user == number:
                        print("test data: " + str(user))
                        print("file: " + file)
                        self.raw_test_data = np.append(self.raw_test_data, new_array, axis=0)

            # print(self.raw_training_data.shape)
            # print(self.raw_test_data.shape)
            print("========================================")
            self.experiment_data = self.raw_training_data
            self.process_data()
            self.training_data = self.data_filtered
            # self.training_data = self.data_norm

            print("========================================")
            self.experiment_data = self.raw_test_data
            self.process_data()
            # self.test_data = self.data_norm
            self.test_data = self.data_filtered

        else:
            print("Could not find learning data file")

        np.save("/tmp/training_data.npy", self.training_data)
        np.save("/tmp/test_data.npy", self.test_data)
        # np.save(ROOT_DIR + "/data_storage/data/processed_learning_data/Rod_learning_data_7.npy", self.test_data)

        print("<======================================================>")
        print("Learning data shape: " + str(self.training_data.shape))
        print("Testing data shape: " + str(self.test_data.shape))
        print("<======================================================>")
        print("Script lasted for: " + str(round((time.time() - st), 2)) + " seconds")

    def process_data(self):

        if self.storage_config["action_classes"] == self.training_config["action_classes"]:
            self.data_classes_filtered = self.experiment_data
        else:
            self.filter_classes(self.experiment_data, self.storage_config, self.training_config)
            print("Class filter complete...")

        if self.storage_config["time"] == self.training_config["time"]:
            self.data_shortened = self.data_classes_filtered
        else:
            self.sample_shortener(self.data_classes_filtered, self.measurements, self.storage_config,
                                  self.training_config)
            # self.sample_shortener(experiment_data, measurements, storage_config, training_config)
            print("Time truncation complete...")

        # if len(storage_config["data"]) == len(training_config["data_filtered"]):
        if len(self.storage_config["data"]) == len(self.training_config["data"]):
            self.data_filtered = self.data_shortened
        else:
            self.filter_data(self.data_shortened, self.storage_config, self.training_config)
            print("Variable filter complete...")

        # self.normalize_data(self.data_filtered, self.storage_config, self.training_config)
        # print("Normalization complete...")

        # np.random.shuffle(self.data_norm)
        np.random.shuffle(self.data_filtered)
        print("Shuffle complete...")

    def sample_shortener(self, array, measurements, store_config, train_config):

        new_measurements = store_config["rate"] * train_config["time"]
        array_shortened = np.empty((0, int(new_measurements * len(store_config["data"]))))

        classification = []
        for vector in array:
            data_array = np.reshape(vector[:-1], (measurements, int(len(vector) / measurements)))
            idx_start = 0
            idx_end = int(new_measurements)

            c = vector[-1]

            # while True:
            #     if idx_end > measurements:
            #         break
            #
            #     data_array[idx_start:idx_end, 0] = data_array[idx_start:idx_end, 0] - data_array[idx_start:idx_end, 0][0]

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
            data_array_norm = np.empty((data_array.shape[0], 0))

            idx = 0
            for n in train_config["normalization_clusters"]:
                data_sub_array = data_array[:, idx:idx+n]
                idx += n

                data_max = abs(max(data_sub_array.min(), data_sub_array.max(), key=abs))

                data_sub_array_norm = data_sub_array / data_max
                data_array_norm = np.hstack((data_array_norm, data_sub_array_norm))

            vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

            # experiment_array_norm = normalize(data_array, axis=0, norm='max')
            #
            # vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))

            array_norm = np.append(array_norm, vector_data_norm, axis=0)

        array_norm = np.append(array_norm, np.reshape([array[:, -1]], (-1, 1)), axis=1)

        self.data_norm = array_norm

    def filter_data(self, array, store_config, train_config):

        measurements = int(store_config["rate"] * train_config["time"])
        data = store_config["data"]
        # data_filtered = train_config["data_filtered"]
        data_filtered = train_config["data"]

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

    def filter_classes(self, array, store_config, train_config):

        class_idx = []

        for i in range(0, len(store_config["action_classes"])):
            if store_config["action_classes"][i] in train_config["action_classes"]:
                class_idx.append(i)

        new_array = np.empty((0, array.shape[1]))

        for j in range(0, array.shape[0]):
            if int(array[j, -1]) in class_idx:
                new_array = np.append(new_array, [array[j, :]], axis=0)

        self.data_classes_filtered = new_array


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for sorter script")
    # parser.add_argument("-p", "--path", type=str, default=ROOT_DIR + "/data_storage/data/raw_learning_data/",
    parser.add_argument("-p", "--path", type=str,
                        default=ROOT_DIR + "/data_storage/data/processed_learning_data/",
                        help="The relative path to the .npy file")
    parser.add_argument("-f", "--file", type=str, default="raw_learning_data.npy",
                        help="The name of the .npy file")
    parser.add_argument("-d", "--div", type=float, default=0.7,
                        help="Percentage of samples that will be used for training and validation (0 to 1). The rest "
                             "will be used for tests")
    parser.add_argument("-c", "--config_file", type=str, default="training_config",
                        help="If argmument is present, activates gripper")

    args = vars(parser.parse_args())

    sort_data_for_learing = SortedDataForLearning(path=args["path"], data_file=args["file"],
                                                  config_file=args["config_file"], div=args["div"])

