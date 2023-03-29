#!/usr/bin/env python3
#
import keras
from numpy import save
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from larcc_classes.documentation.PDF import PDF
from neural_networks.utils import plot_confusion_matrix_percentage, prediction_classification, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay
from progressbar import progressbar

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np


if __name__ == '__main__':

    sliding_window = 20
    time_steps = 100
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    print("time_steps-sliding_window+1")
    print(time_steps-sliding_window+1)

    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    sorted_data_for_learning = SortedDataForLearning(path=path)

    test_data = sorted_data_for_learning.test_data
    n_test = test_data.shape[0]

    cnn_model = keras.models.load_model("cnn4_model_20ms")

    x_test_cnn = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13, 1))
    y_test = test_data[:, -1]

    # CONVOLUTIONAL TESTING
    pred_cnn = []
    for i in progressbar(range(len(x_test_cnn)), redirect_stdout=True):
    # for i in range(0, len(x_test_cnn)):
        sample_pred = []
        for sw in range(0, time_steps-sliding_window+1):
            prediction = cnn_model.predict(x=x_test_cnn[i:i+1, sw:sw+sliding_window, :, :], verbose=2)
            sample_pred.append(prediction)

        pred_cnn.append(sample_pred)

    pred_cnn = np.array(pred_cnn)
    print("\n")
    print("pred_cnn.shape")
    print(pred_cnn.shape)
    print("\n")
    pred_cnn = np.reshape(pred_cnn, (pred_cnn.shape[0], pred_cnn.shape[1], pred_cnn.shape[3]))

    save('cnn4_20ms_data2_pred.npy', pred_cnn)

    # TRUE RESULTS
    y_test_final = []
    for line in range(0, y_test.shape[0], 2):
        r = [int(y_test[line]), int(y_test[line + 1])]
        y_test_final.append(r)
    save('true_results_data2.npy', y_test_final[0:len(x_test_cnn)])