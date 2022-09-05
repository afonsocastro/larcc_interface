#!/usr/bin/env python3
import json
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import statistics

path = "../data/sensor_testing/paper_analysis/"

res = os.listdir(path)
i = 0

for file in res:
    print(f'[{i}]:' + file)
    i += 1

idx = input("Select idx from test json: ")

f = open(path + res[int(idx)])

data = json.load(f)

f.close()
