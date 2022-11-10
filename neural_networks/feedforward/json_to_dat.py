#!/usr/bin/env python3
import json
import os

import numpy as np

from config.definitions import ROOT_DIR


with open(ROOT_DIR + "/neural_networks/feedforward/training_testing_n_times/training_curves_mean.json") as file:
    total_data = json.load(file)

# for key, value in total_data.items():
#     print(key)

indices = list(range(0, len(total_data["loss"])))

loss = total_data["loss"]
accuracy = total_data["accuracy"]
val_loss = total_data["val_loss"]
val_accuracy = total_data["val_accuracy"]

save_path = ROOT_DIR + "/neural_networks/feedforward/training_testing_n_times/"

np.savetxt(save_path + 'loss.dat', np.column_stack((indices, loss)), fmt=['%.0f', '%.5f'])
np.savetxt(save_path + 'accuracy.dat', np.column_stack((indices, accuracy)), fmt=['%.0f', '%.5f'])
np.savetxt(save_path + 'val_loss.dat', np.column_stack((indices, val_loss)), fmt=['%.0f', '%.5f'])
np.savetxt(save_path + 'val_accuracy.dat', np.column_stack((indices, val_accuracy)), fmt=['%.0f', '%.5f'])



