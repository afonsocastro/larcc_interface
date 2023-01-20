#!/usr/bin/env python3
import keras.models
import matplotlib.pyplot as plt
from keras import Input, Model
import tensorflow as tf
from keras.callbacks import EarlyStopping
# from keras.utils import to_categorical  # one-hot encode target column
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
from keras.layers import Dense, LSTM, Layer
from keras.utils.vis_utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from progressbar import progressbar
from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np
from neural_networks.utils import plot_confusion_matrix_percentage
from keras.layers import Lambda
from keras import backend as K
from numpy import save, delete


if __name__ == '__main__':
    validation_split = 0.3
    batch_size = 2
    # time_steps = 50
    time_steps = 100
    neurons = 16
    params = 12
    labels = 4
    start_number = 17
    epochs = 50
    # epochs = 50
    time_window = 2
    input_nn = time_window * 10

    # delete('pred_model_Bahdanau_Attention.npy')
    # delete('true_model_Bahdanau_Attention.npy')

    # config_file = "training_config_time_" + str(time_window)

    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    # sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)
    sorted_data_for_learning = SortedDataForLearning(path=path)

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    n_test = test_data.shape[0]

    cnn4_model = keras.models.load_model(ROOT_DIR + "/neural_networks/convolutional/cnn4_model_02_time_window")

    x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
    y_test = test_data[:, -1]
    x_test_rnn = x_test[:, :, 1:]
    x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    print("len(x_test)")
    print(len(x_test))
    print("x_test.shape")
    print(x_test.shape)

    pred_cnn = np.array()
    # for i in range(0, len(x_test)-1):
    for i in range(0, 2):
        sample_pred = np.array()
        for sw in range(0, 80):
            prediction = cnn4_model.predict(x=x_test_cnn[0:1, sw:sw+20, :, :], verbose=0)
            np.append(sample_pred, prediction)
            # sample_pred.append(prediction)
            # print("prediction")
            # print(prediction)
        np.append(pred_cnn, sample_pred)
        # pred_cnn.append(sample_pred)
        print("pred_cnn")
        print(pred_cnn)
        print("type(pred_cnn)")
        print(type(pred_cnn))
        print("pred_cnn.shape")
        print(pred_cnn.shape)

    # print("type(pred_cnn)")
    # print(type(pred_cnn))
    # print("pred_cnn.shape")
    # print(pred_cnn.shape)


