#!/usr/bin/env python3
import json
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import statistics

path = "../../data/sensor_testing/initial_pose_testing/"

res = os.listdir(path)

tests = ['100ms', '200ms', '500ms', '1000ms', '10s']

labels = ["fx", "fy", "fz", "mx", "my", "mz"]

for label in labels:
    for test in tests:

        list_tests = [s for s in res if test in s]

        print("time: " + test)
        for experiment in list_tests:
            f = open(path + experiment)

            data = json.load(f)

            f.close()

            print(f"{label} mean: " + str(round(data[label]["mean"], 5)))



