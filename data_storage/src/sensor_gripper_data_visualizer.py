#!/usr/bin/env python3
import json
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import statistics

path = "../data/sensor_testing/"

res = os.listdir(path)
i = 0

for file in res:
    print(f'[{i}]:' + file)
    i += 1

idx = input("Select idx from test json: ")

f = open(path + f'test{idx}.json')

data = json.load(f)

f.close()

labels = ["fx", "fy", "fz", "mx", "my", "mz"]
hist_width = [0.025, 0.0005]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(0, 2):
    for j in range(0, 3):
        idx = j + 3 * i
        mean = data[labels[idx]]["mean"]
        values = data[labels[idx]]["values"]

        low = mean - data[labels[idx]]["confidence_interval"]
        high = mean + data[labels[idx]]["confidence_interval"]

        x = np.arange(min(values), max(values), hist_width[i])
        axs[i, j].hist(values, bins=x)
        axs[i, j].title.set_text(labels[idx])
        axs[i, j].plot([mean, mean], [0, 30], color='red')
        axs[i, j].plot([low, low], [0, 30], color='yellow')
        axs[i, j].plot([high, high], [0, 30], color='yellow')

plt.show()
