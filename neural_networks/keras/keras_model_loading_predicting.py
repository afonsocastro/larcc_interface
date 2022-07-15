#!/usr/bin/env python3

import numpy as np
from tensorflow import keras


if __name__ == '__main__':

    all_data = np.load('../data/learning_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    model = keras.models.load_model("myModel")

    predictions = model.predict(x=all_data[27:28, :-1], verbose=2)

    print("\n")
    print("predictions")
    print(predictions)

    print("\n")

    print("all_data[27:28, -1]")
    print(all_data[27:28, -1])
    