#!/usr/bin/env python3
import json
import os

import numpy as np

from config.definitions import ROOT_DIR

# file = ROOT_DIR + "/data_storage/data/predicted_learning_data/multi_class_sample_1.json"

file = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/Ru_learning_data_5.npy"
#
# f = open(file)
#
# json_file = json.load(f)
#
# f.close()
#
# data = np.array(json_file["data_predicted"])[:, :-1]

data_file = np.load(file)
data = data_file[:, :-1]

time_max_prev = 0

time = []

fx = []
fy = []
fz = []

mx = []
my = []
mz = []

j0 = []
j1 = []
j2 = []
j3 = []
j4 = []
j5 = []

time_divisions = []

i = 164
# for i in range(data.shape[0]):

data_vector = data[i, :]
data_array = np.reshape(data_vector, (50, int(len(data_vector) / 50)))
print(i)

new_time = data_array[:, 0] + [time_max_prev] * 50

time.extend(new_time)

time_max_prev = time[-1]
time_divisions.append(time_max_prev)

j0.extend(data_array[:, 1])
j1.extend(data_array[:, 2])
j2.extend(data_array[:, 3])
j3.extend(data_array[:, 4])
j4.extend(data_array[:, 5])
j5.extend(data_array[:, 6])

fx.extend(data_array[:, 7])
fy.extend(data_array[:, 8])
fz.extend(data_array[:, 9])

mx.extend(data_array[:, 10])
my.extend(data_array[:, 11])
mz.extend(data_array[:, 12])

print(time_divisions)

save_path = ROOT_DIR + "/data_storage/data/predicted_learning_data/"

np.savetxt(save_path + 'fx_normal.dat', np.column_stack((time, fx)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'fy_normal.dat', np.column_stack((time, fy)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'fz_normal.dat', np.column_stack((time, fz)), fmt=['%.3f', '%.3f'])

np.savetxt(save_path + 'mx_normal.dat', np.column_stack((time, mx)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'my_normal.dat', np.column_stack((time, my)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'mz_normal.dat', np.column_stack((time, mz)), fmt=['%.3f', '%.3f'])

np.savetxt(save_path + 'j1_normal.dat', np.column_stack((time, j0)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'j2_normal.dat', np.column_stack((time, j1)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'j3_normal.dat', np.column_stack((time, j2)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'j4_normal.dat', np.column_stack((time, j3)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'j5_normal.dat', np.column_stack((time, j4)), fmt=['%.3f', '%.3f'])
np.savetxt(save_path + 'j6_normal.dat', np.column_stack((time, j5)), fmt=['%.3f', '%.3f'])




# path = ROOT_DIR + "/data_storage/data/sensor_testing/paper_analysis/second_attempt/"
#
# res = os.listdir(path)
# i = 0
#
# for file in res:
#     print(f'[{i}]:' + file)
#     i += 1
#
# idx = input("Select idx from test json: ")
#
# f = open(path + res[int(idx)])
#
# data = json.load(f)
#
# f.close()
#
# fx = np.array(data["fx"])
# fy = np.array(data["fy"])
# fz = np.array(data["fz"])
#
# timestamp = np.array(data["timestamp"])
# print(timestamp.shape)
# print(fx.shape)
# print(zip(timestamp, fx))
#
# print(np.column_stack((timestamp, fx)))
#
# np.savetxt('fx_output.dat', np.column_stack((timestamp, fx)), fmt=['%.3f', '%.3f'])
# np.savetxt('fy_output.dat', np.column_stack((timestamp, fy)), fmt=['%.3f', '%.3f'])
# np.savetxt('fz_output.dat', np.column_stack((timestamp, fz)), fmt=['%.3f', '%.3f'])
