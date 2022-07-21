#!/usr/bin/env python3

from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    all_data = np.load('../data/learning_data_test.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    model = keras.models.load_model("myModel")
    predictions_list = []
    # predictions = model.predict(x=all_data[27:28, :-1], verbose=2)

    for i in range(0, len(all_data)):
        prediction = model.predict(x=all_data[i:i+1, :-1], verbose=2)
        decoded_prediction = np.argmax(prediction)
        predictions_list.append(decoded_prediction)

    predicted_values = np.asarray(predictions_list)

    cm = confusion_matrix(y_true=all_data[:, -1], y_pred=predicted_values)

    cm_plot_labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_plot_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()
