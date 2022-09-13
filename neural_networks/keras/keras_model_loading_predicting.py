#!/usr/bin/env python3

from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from config.definitions import ROOT_DIR


if __name__ == '__main__':

    test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    model = keras.models.load_model("myModel")
    predictions_list = []
    pull_matx = []
    push_matx = []
    shake_matx = []

    col = test_data.shape[1]
    predicted_pull_matx = np.empty((0, col))
    predicted_push_matx = np.empty((0, col))
    predicted_shake_matx = np.empty((0, col))

    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i+1, :-1], verbose=2)
        predict_lst = [prediction[0][0], prediction[0][1], prediction[0][2]]
        decoded_prediction = np.argmax(prediction)

        if decoded_prediction == 0:
            pull_matx.append(predict_lst)
            predicted_pull_matx = np.append(predicted_pull_matx, [test_data[i, :]], axis=0)

        elif decoded_prediction == 1:
            push_matx.append(predict_lst)
            predicted_push_matx = np.append(predicted_push_matx, [test_data[i, :]], axis=0)

        elif decoded_prediction == 2:
            shake_matx.append(predict_lst)
            predicted_shake_matx = np.append(predicted_shake_matx, [test_data[i, :]], axis=0)

        predictions_list.append(decoded_prediction)

    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_pull.npy", predicted_pull_matx)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_push.npy", predicted_push_matx)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_shake.npy", predicted_shake_matx)

    pull_mat = np.matrix(predicted_pull_matx)
    with open(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_pull.txt", 'wb') as f:
        for line in pull_mat:
            np.savetxt(f, line, fmt='%.2f')

    push_mat = np.matrix(predicted_push_matx)
    with open(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_push.txt", 'wb') as f:
        for line in push_mat:
            np.savetxt(f, line, fmt='%.2f')

    shake_mat = np.matrix(predicted_shake_matx)
    with open(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_shake.txt", 'wb') as f:
        for line in shake_mat:
            np.savetxt(f, line, fmt='%.2f')

    pull_matrix = np.array(pull_matx)
    push_matrix = np.array(push_matx)
    shake_matrix = np.array(shake_matx)

    # Calculate mean values across each column
    mean_0 = pull_matrix.mean(axis=0)
    minimum_0 = np.min(pull_matrix, axis=0)
    maximum_0 = np.max(pull_matrix, axis=0)

    mean_1 = push_matrix.mean(axis=0)
    minimum_1 = np.min(push_matrix, axis=0)
    maximum_1 = np.max(push_matrix, axis=0)

    mean_2 = shake_matrix.mean(axis=0)
    minimum_2 = np.min(shake_matrix, axis=0)
    maximum_2 = np.max(shake_matrix, axis=0)

    labels = ['PULL', 'PUSH', 'SHAKE']

    # data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
    # names = list(data.keys())
    # values = list(data.values())

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    p0 = axs[0].bar(labels, mean_0)
    axs[0].set_title("predicted PUSHES")

    # p0.label(mean_0)

    axs[1].bar(labels, mean_1)
    axs[1].set_title("predicted PULLS")
    # for i, v in enumerate(mean_1):
    #     axs[1].text(v -0.5, i -0.5, str(v), color='black', fontweight='bold')

    axs[2].bar(labels, mean_2)
    axs[2].set_title("predicted SHAKES")

    # axs[2].bar_label(mean_2, padding=3)

    fig.suptitle('Output Confidence')

    # plt.show()
    # exit(0)

    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confidence_histogram.png", bbox_inches='tight')

    predicted_values = np.asarray(predictions_list)

    cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)

    # cm_plot_labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    cm_plot_labels = ['PULL', 'PUSH', 'SHAKE']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_plot_labels)

    disp.plot(cmap=plt.cm.Blues)
    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confusion_matrix.png", bbox_inches='tight')
