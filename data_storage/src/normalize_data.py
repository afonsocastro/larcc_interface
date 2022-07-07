import numpy as np
from sklearn.preprocessing import normalize


path = "./../data/trainning/"

data = np.load(path + "trainning_data.npy")

num_values = 28

data_norm = np.empty((0, data.shape[1]), dtype=float)


for experiment in data:
    experiment_array = np.reshape(experiment, (int(len(experiment) / num_values), num_values))
    experiment_array_norm = normalize(experiment_array, axis=0, norm='max')

    row_norm = np.reshape(experiment_array_norm, (1, 840))

    data_norm = np.append(data_norm, row_norm, axis=0)


np.save(path + "trainning_data_norm.npy", data_norm)

