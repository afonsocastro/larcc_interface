#!/usr/bin/env python3

from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from config.definitions import ROOT_DIR
from tabulate import tabulate
import pyfiglet
from colorama import Fore


def valuelabel(ax, labelss, values, pose, pred):
    for i in range(len(labelss)):
        if pred == i:
            ax.text(i, values[i] + pose, round(values[i], 5), ha='center')
        else:
            ax.text(i, values[i] + pose, round(values[i], 5), ha='center')


if __name__ == '__main__':

    test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    # test_data = np.load(ROOT_DIR + '/data_storage/data/trainning_learning_data/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
    #                     encoding='ASCII')

    model = keras.models.load_model("myModel")

    n_labels = 3

    predictions_list = []
    pull_matx = []
    push_matx = []
    shake_matx = []
    # twist_matx = []

    col = test_data.shape[1]
    predicted_pull_matx = np.empty((0, col))
    predicted_push_matx = np.empty((0, col))
    predicted_shake_matx = np.empty((0, col))
    # predicted_twist_matx = np.empty((0, col))

    output_predicted_pull = np.empty((0, n_labels))
    output_predicted_push = np.empty((0, n_labels))
    output_predicted_shake = np.empty((0, n_labels))
    # output_predicted_twist = np.empty((0, n_labels))

    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i+1, :-1], verbose=2)
        predict_lst = [prediction[0][0], prediction[0][1], prediction[0][2]]
        decoded_prediction = np.argmax(prediction)

        if decoded_prediction == 0:
            pull_matx.append(predict_lst)
            predicted_pull_matx = np.append(predicted_pull_matx, [test_data[i, :]], axis=0)
            output_predicted_pull = np.append(output_predicted_pull, prediction, axis=0)

        elif decoded_prediction == 1:
            push_matx.append(predict_lst)
            predicted_push_matx = np.append(predicted_push_matx, [test_data[i, :]], axis=0)
            output_predicted_push = np.append(output_predicted_push, prediction, axis=0)

        elif decoded_prediction == 2:
            shake_matx.append(predict_lst)
            predicted_shake_matx = np.append(predicted_shake_matx, [test_data[i, :]], axis=0)
            output_predicted_shake = np.append(output_predicted_shake, prediction, axis=0)

        # elif decoded_prediction == 3:
        #     twist_matx.append(predict_lst)
        #     predicted_twist_matx = np.append(predicted_twist_matx, [test_data[i, :]], axis=0)
        #     output_predicted_twist = np.append(output_predicted_twist, prediction, axis=0)

        predictions_list.append(decoded_prediction)

    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_pull.npy", predicted_pull_matx)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_push.npy", predicted_push_matx)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_shake.npy", predicted_shake_matx)
    # np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_twist.npy", predicted_twist_matx)

    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_pull.npy", output_predicted_pull)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_push.npy", output_predicted_push)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_shake.npy", output_predicted_shake)
    # np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_twist.npy", output_predicted_twist)

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

    output_pull_mat = np.matrix(output_predicted_pull)
    with open(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_pull.txt", 'wb') as f:
        for line in output_pull_mat:
            np.savetxt(f, line, fmt='%.2f')

    output_push_mat = np.matrix(output_predicted_push)
    with open(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_push.txt", 'wb') as f:
        for line in output_push_mat:
            np.savetxt(f, line, fmt='%.2f')

    output_shake_mat = np.matrix(output_predicted_shake)
    with open(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_shake.txt", 'wb') as f:
        for line in output_shake_mat:
            np.savetxt(f, line, fmt='%.2f')

    pull_matrix = np.array(pull_matx)
    push_matrix = np.array(push_matx)
    shake_matrix = np.array(shake_matx)

    # Calculate mean values across each column
    mean_0 = pull_matrix.mean(axis=0)
    std_0 = pull_matrix.std(axis=0)
    minimum_0 = np.min(pull_matrix, axis=0)
    maximum_0 = np.max(pull_matrix, axis=0)

    mean_1 = push_matrix.mean(axis=0)
    std_1 = push_matrix.std(axis=0)
    minimum_1 = np.min(push_matrix, axis=0)
    maximum_1 = np.max(push_matrix, axis=0)

    mean_2 = shake_matrix.mean(axis=0)
    std_2 = shake_matrix.std(axis=0)
    minimum_2 = np.min(shake_matrix, axis=0)
    maximum_2 = np.max(shake_matrix, axis=0)

    labels = ['PULL', 'PUSH', 'SHAKE']

    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    axs[0].bar(labels, mean_0)
    axs[0].set_title("predicted PULLS (" + str(len(pull_matrix)) + ")")

    axs[0].text(0, mean_0[0] -0.05, round(mean_0[0], 5), ha='center', fontweight='bold')
    axs[0].text(1, mean_0[1] + 0.1, round(mean_0[1], 5), ha='center', fontweight='bold')
    axs[0].text(2, mean_0[2] + 0.1, round(mean_0[2], 5), ha='center', fontweight='bold')

    axs[0].text(0, mean_0[0] - 0.1, round(std_0[0], 5), ha='center', style='italic')
    axs[0].text(1, mean_0[1] + 0.05, round(std_0[1], 5), ha='center', style='italic')
    axs[0].text(2, mean_0[2] + 0.05, round(std_0[2], 5), ha='center', style='italic')

    axs[0].text(0, minimum_0[0] - 0.03, round(minimum_0[0], 5), ha='center')
    # axs[0].text(1, minimum_0[1] - 0.03, round(minimum_0[1], 5), ha='center')
    # axs[0].text(2, minimum_0[2] - 0.03, round(minimum_0[2], 5), ha='center')

    for i in range(0, n_labels):
        if maximum_0[i] > 0.15:
            axs[0].text(i, maximum_0[i] + 0.01, round(maximum_0[i], 5), ha='center')

    # axs[0].text(1, maximum_0[1] + 0.01, round(maximum_0[1], 5), ha='center')
    # axs[0].text(2, maximum_0[2] + 0.01, round(maximum_0[2], 5), ha='center')
    # valuelabel(axs[0], labels, minimum_0, 0.01, 0)
    # valuelabel(axs[0], labels, maximum_0, 0.01, 0)

    for p in axs[0].patches:
        w = p.get_width()  # get width of bar
        axs[0].hlines(minimum_0[0], -w/2, w/2, colors='r')
        axs[0].hlines(minimum_0[1], 0.2 + w/2, 0.2 + w/2 + w, colors='r')
        axs[0].hlines(minimum_0[2], w/2 + 0.4 + w, w/2 + 0.4 + w + w, colors='r')

        axs[0].hlines(maximum_0[0], -w / 2, w / 2, colors='g')
        axs[0].hlines(maximum_0[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        axs[0].hlines(maximum_0[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')


    axs[1].bar(labels, mean_1)
    # axs[1].set_title("predicted PUSHES")
    axs[1].set_title("predicted PUSHES (" + str(len(push_matrix)) + ")")

    axs[1].text(0, mean_1[0] + 0.1, round(mean_1[0], 5), ha='center', fontweight='bold')
    axs[1].text(1, mean_1[1] - 0.05, round(mean_1[1], 5), ha='center', fontweight='bold')
    axs[1].text(2, mean_1[2] + 0.1, round(mean_1[2], 5), ha='center', fontweight='bold')

    axs[1].text(0, mean_1[0] + 0.05, round(std_1[0], 5), ha='center', style='italic')
    axs[1].text(1, mean_1[1] - 0.1, round(std_1[1], 5), ha='center', style='italic')
    axs[1].text(2, mean_1[2] + 0.05, round(std_1[2], 5), ha='center', style='italic')

    # axs[1].text(0, minimum_1[0] - 0.03, round(minimum_1[0], 5), ha='center')
    axs[1].text(1, minimum_1[1] - 0.03, round(minimum_1[1], 5), ha='center')
    # axs[1].text(2, minimum_1[2] - 0.03, round(minimum_1[2], 5), ha='center')

    for i in range(0, n_labels):
        if maximum_1[i] > 0.15:
            axs[1].text(i, maximum_1[i] + 0.01, round(maximum_1[i], 5), ha='center')

    # axs[1].text(0, maximum_1[0] + 0.01, round(maximum_1[0], 5), ha='center')
    # axs[1].text(1, maximum_1[1] + 0.01, round(maximum_1[1], 5), ha='center')
    # axs[1].text(2, maximum_1[2] + 0.01, round(maximum_1[2], 5), ha='center')

    for p in axs[1].patches:
        w = p.get_width()  # get width of bar
        axs[1].hlines(minimum_1[0], -w/2, w/2, colors='r')
        axs[1].hlines(minimum_1[1], 0.2 + w/2, 0.2 + w/2 + w, colors='r')
        axs[1].hlines(minimum_1[2], w/2 + 0.4 + w, w/2 + 0.4 + w + w, colors='r')

        axs[1].hlines(maximum_1[0], -w / 2, w / 2, colors='g')
        axs[1].hlines(maximum_1[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        axs[1].hlines(maximum_1[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')

    axs[2].bar(labels, mean_2)
    # axs[2].set_title("predicted SHAKES")
    axs[2].set_title("predicted SHAKES (" + str(len(shake_matrix)) + ")")

    axs[2].text(0, mean_2[0] + 0.1, round(mean_2[0], 5), ha='center', fontweight='bold')
    axs[2].text(1, mean_2[1] + 0.1 , round(mean_2[1], 5), ha='center', fontweight='bold')
    axs[2].text(2, mean_2[2] - 0.05, round(mean_2[2], 5), ha='center', fontweight='bold')

    axs[2].text(0, mean_2[0] + 0.05, round(std_2[0], 5), ha='center', style='italic')
    axs[2].text(1, mean_2[1] + 0.05, round(std_2[1], 5), ha='center', style='italic')
    axs[2].text(2, mean_2[2] - 0.1, round(std_2[2], 5), ha='center', style='italic')

    # axs[2].text(0, minimum_2[0] - 0.03, round(minimum_2[0], 5), ha='center')
    # axs[2].text(1, minimum_2[1] - 0.03, round(minimum_2[1], 5), ha='center')
    axs[2].text(2, minimum_2[2] - 0.03, round(minimum_2[2], 5), ha='center')

    for i in range(0, n_labels):
        if maximum_2[i] > 0.15:
            axs[2].text(i, maximum_2[i] + 0.01, round(maximum_2[i], 5), ha='center')

    # axs[2].text(0, maximum_2[0] + 0.01, round(maximum_2[0], 5), ha='center')
    # axs[2].text(1, maximum_2[1] + 0.01, round(maximum_2[1], 5), ha='center')
    # axs[2].text(2, maximum_2[2] + 0.01, round(maximum_2[2], 5), ha='center')

    for p in axs[2].patches:
        w = p.get_width()  # get width of bar
        axs[2].hlines(minimum_2[0], -w / 2, w / 2, colors='r')
        axs[2].hlines(minimum_2[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='r')
        axs[2].hlines(minimum_2[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='r')

        axs[2].hlines(maximum_2[0], -w / 2, w / 2, colors='g')
        axs[2].hlines(maximum_2[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        axs[2].hlines(maximum_2[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')

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

    total_pull = cm[0][0] + cm[1][0] + cm[2][0]
    pull_acc = cm[0][0] / total_pull

    total_push = cm[0][1] + cm[1][1] + cm[2][1]
    push_acc = cm[1][1] / total_push

    total_shake = cm[0][2] + cm[1][2] + cm[2][2]
    shake_acc = cm[2][2] / total_shake

    total = sum(sum(cm))

    print("total")
    print(total)

    total_right = cm[0][0] + cm[1][1] + cm[2][2]

    print("total_right")
    print(total_right)
    # fig2 = plt.plot()
    accs = [pull_acc, push_acc, shake_acc]
    columns = ('PULL', 'PUSH', 'SHAKE')
    rows = ["accuracy"]

    print("\n")

    print(Fore.LIGHTBLUE_EX + "Confusion Matrix Accuracy" + Fore.RESET)
    print(tabulate([accs], headers=cm_plot_labels, tablefmt="fancy_grid"))
    print("\n")
    print(Fore.LIGHTBLUE_EX + "Total Accuracy: " + Fore.LIGHTYELLOW_EX + str(round(total_right/total, 5)) + Fore.RESET)
    print("\n")

    # plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/table.png", bbox_inches='tight')
