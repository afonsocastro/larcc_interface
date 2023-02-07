#!/usr/bin/env python3
import json

from config.definitions import ROOT_DIR
import numpy as np


def print_stats(u, array, actions):
    classifications = array[:, -1]
    class_len = {}

    for c in classifications:
        if c in class_len:
            class_len[c] += 1
        else:
            class_len[c] = 1

    print(f"User {u}")
    print(f"User has {array.shape[0]} experiments")

    for c_l in class_len:
        print(f"{actions[int(c_l)]}:{class_len[c_l]}")

    print("===============================================")
    print("===============================================")

if __name__ == '__main__':

    raw_data = np.load(ROOT_DIR + "/data_storage/data/new_acquisition/raw_learning_data.npy")

    f = open(ROOT_DIR + '/data_storage/config/user_split_new.json')

    split_config = json.load(f)

    f.close()

    dic_of_matrixs = {}

    for i in range(0, len(split_config["users"])):
        dic_of_matrixs[f"{i+6}"] = np.empty((0, raw_data.shape[1]))

    idx_start = 0
    for j, user in enumerate(split_config["user_order"]):
        idx_end = idx_start + split_config["user_idxs"][j]
        dic_of_matrixs[f"{user}"] = np.append(dic_of_matrixs[f"{user}"],
                                              raw_data[idx_start:idx_end, :], axis=0)

        idx_start += split_config["user_idxs"][j]

    users = split_config["users"]
    for idx in dic_of_matrixs:
        np.save(ROOT_DIR + f"/data_storage/data/new_acquisition/user_splitted_data/"
                           f"{users[int(idx)-6]}_learning_data_{int(idx)}.npy", dic_of_matrixs[idx])

        print_stats(users[int(idx)-6], dic_of_matrixs[idx], split_config["actions"])



