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

    config_file = "training_config_time_" + str(time_window)

    path = ROOT_DIR + "/data_storage/data/raw_learning_data/user_splitted_data/"
    sorted_data_for_learning = SortedDataForLearning(path=path, config_file=config_file)

    training_data = sorted_data_for_learning.training_data
    test_data = sorted_data_for_learning.test_data

    n_train = len(training_data) * (1 - validation_split)
    n_val = len(training_data) * validation_split
    n_test = test_data.shape[0]

    cnn4_model = keras.models.load_model(ROOT_DIR + "/neural_networks/convolutional/cnn4_model_02_time_window")

    for i in range(0, len(test_data)):
        x_test = np.reshape(test_data[i:i + 1, :-1], (1, input_nn, 13, 1))
        prediction = cnn4_model.predict(x=x_test, verbose=0)

        # Reverse to_categorical from keras utils
        decoded_prediction = np.argmax(prediction, axis=1, out=None)

        true = test_data[i, -1]

        print('expected: ', true)
        print('predicted: ', decoded_prediction)


    # x_test = np.reshape(test_data[:, :-1], (int(n_test / 2), time_steps, 13))
    # y_test = test_data[:, -1]
    # x_test = x_test[:, :, 1:]
    #
    # y_test_final = []
    # for line in range(0, y_test.shape[0], 2):
    #     r = [int(y_test[line]), int(y_test[line + 1])]
    #     y_test_final.append(r)
    #
    # pred = model_Bahdanau_Attention.predict(x_test[0:260].reshape(260, time_steps, params), batch_size=2)
    #
    # save('pred_model_Bahdanau_Attention.npy', pred)
    # save('true_model_Bahdanau_Attention.npy', y_test_final[0:260])

    # print('input: ', x_test[0:2].shape)
    # print('expected: ', y_test_final)
    # print('predicted: ', pred[0:2])
