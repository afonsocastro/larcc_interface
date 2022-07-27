#!/usr/bin/env python3
import json
import math
import os
import time
import rospy
from colorama import Fore
import numpy as np
from matplotlib import pyplot as plt
from data_aquisition_node import DataForLearning
from scipy.stats import norm
import statistics


def count(list1, low, up):
    c = 0
    # traverse in the list1
    for x in list1:
        # condition check
        if low <= x <= up:
            c += 1
    return c


rospy.init_node("test_aquisition", anonymous=True)

rate = rospy.Rate(100)

data_for_learning = DataForLearning()

time.sleep(0.2)

i = 0

data = {"fx": [],
        "fy": [],
        "fz": [],
        "mx": [],
        "my": [],
        "mz": []}


st = time.time()
while not rospy.is_shutdown():
    print(data_for_learning)

    current_data = data_for_learning

    data["fx"].append(current_data.wrench_force_torque.force.x)
    data["fy"].append(current_data.wrench_force_torque.force.y)
    data["fz"].append(current_data.wrench_force_torque.force.z)
    data["mx"].append(current_data.wrench_force_torque.torque.x)
    data["my"].append(current_data.wrench_force_torque.torque.y)
    data["mz"].append(current_data.wrench_force_torque.torque.z)

    if i > 300:
        break

    i += 1
    rate.sleep()

et = time.time()

struct = {"duration": et - st,
          "joint_position": data_for_learning.joints_position}

for key in data:
    val_min = str(min(data[key]))
    val_max = str(max(data[key]))
    val_len = len(data[key])
    val_sum = sum(data[key])
    val_average = float(val_sum / val_len)

    deviations = [(x - val_average) ** 2 for x in data[key]]

    variance = sum(deviations) / val_len

    SD = math.sqrt(variance)

    CI = 3.29 * (SD / math.sqrt(val_len))

    print(Fore.BLUE + key + ":" + Fore.RESET + f"\nMean: {val_average}\nSD: {SD}\nCI: "
                                               f"{CI}\nMax: {val_max}\nMin: {val_min}")

    struct[key] = {"values": data[key],
                   "mean": val_average,
                   "standard_deviation": SD,
                   "confidence_interval": CI,
                   "max": float(val_max),
                   "min": float(val_min)}


path = "../data/sensor_testing/initial_pose_testing/"

res = os.listdir(path)

idx = len(res)

with open(path + f'initial_pose_3000ms_{idx + 1 - 43}.json', 'w') as fp:
    json.dump(struct, fp)

x = np.arange(min(data["fx"]), max(data["fx"]), 0.025)
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(data["fx"], bins=x)
plt.show()


