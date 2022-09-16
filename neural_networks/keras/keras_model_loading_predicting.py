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


def save_txt_file(matrix, action, output):

    if not output:
        mat = np.matrix(matrix)
        with open(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_" + action + ".txt", 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')

    if output:
        output_mat = np.matrix(matrix)
        with open(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_" + action + ".txt", 'wb') as output_f:
            for output_line in output_mat:
                np.savetxt(output_f, output_line, fmt='%.2f')


if __name__ == '__main__':

    # labels = ['PULL', 'PUSH', 'SHAKE']
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                        encoding='ASCII')

    # test_data = np.load(ROOT_DIR + '/data_storage/data/trainning_learning_data/test_data.npy', mmap_mode=None,
    # allow_pickle=False, fix_imports=True, encoding='ASCII')

    model = keras.models.load_model("myModel")

    predictions_list = []
    pull_matx = []
    push_matx = []
    shake_matx = []
    twist_matx = []

    col = test_data.shape[1]
    predicted_pull_matx = np.empty((0, col))
    predicted_push_matx = np.empty((0, col))
    predicted_shake_matx = np.empty((0, col))
    predicted_twist_matx = np.empty((0, col))

    output_predicted_pull = np.empty((0, n_labels))
    output_predicted_push = np.empty((0, n_labels))
    output_predicted_shake = np.empty((0, n_labels))
    output_predicted_twist = np.empty((0, n_labels))

    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=2)

        predict_lst = []
        for label in range(0, n_labels):
            predict_lst.append(prediction[0][label])

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

        elif decoded_prediction == 3:
            twist_matx.append(predict_lst)
            predicted_twist_matx = np.append(predicted_twist_matx, [test_data[i, :]], axis=0)
            output_predicted_twist = np.append(output_predicted_twist, prediction, axis=0)

        predictions_list.append(decoded_prediction)

    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_pull.npy", predicted_pull_matx)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_push.npy", predicted_push_matx)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_shake.npy", predicted_shake_matx)

    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_pull.npy", output_predicted_pull)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_push.npy", output_predicted_push)
    np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_shake.npy", output_predicted_shake)

    save_txt_file(predicted_pull_matx, "pull", False)
    save_txt_file(predicted_push_matx, "push", False)
    save_txt_file(predicted_shake_matx, "shake", False)

    save_txt_file(output_predicted_pull, "pull", True)
    save_txt_file(output_predicted_push, "push", True)
    save_txt_file(output_predicted_shake, "shake", True)

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

    mean_3, std_3, minimum_3, maximum_3 = 0, 0, 0, 0

    if n_labels == 4:
        np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/predicted_twist.npy", predicted_twist_matx)
        np.save(ROOT_DIR + "/neural_networks/keras/predicted_data/output_predicted_twist.npy", output_predicted_twist)
        save_txt_file(predicted_twist_matx, "twist", False)
        save_txt_file(output_predicted_twist, "twist", True)
        twist_matrix = np.array(twist_matx)
        mean_3 = twist_matrix.mean(axis=0)
        std_3 = twist_matrix.std(axis=0)
        minimum_3 = np.min(twist_matrix, axis=0)
        maximum_3 = np.max(twist_matrix, axis=0)

    decimal_round = 3

    # fig, axs = plt.subplots(1, n_labels, figsize=(12, 5), sharey=True)
    fig, axs = plt.subplots(1, n_labels, figsize=(18, 6), sharey=True)
    axs[0].bar(labels, mean_0)
    axs[0].set_title("predicted PULLS (" + str(len(pull_matrix)) + ")")

    axs[0].text(0, mean_0[0] - 0.05, round(mean_0[0], decimal_round), ha='center', fontweight='bold')
    axs[0].text(1, mean_0[1] + 0.1, round(mean_0[1], decimal_round), ha='center', fontweight='bold')
    axs[0].text(2, mean_0[2] + 0.1, round(mean_0[2], decimal_round), ha='center', fontweight='bold')
    axs[0].text(3, mean_0[3] + 0.1, round(mean_0[3], decimal_round), ha='center', fontweight='bold')

    axs[0].text(0, mean_0[0] - 0.1, round(std_0[0], decimal_round), ha='center', style='italic')
    axs[0].text(1, mean_0[1] + 0.05, round(std_0[1], decimal_round), ha='center', style='italic')
    axs[0].text(2, mean_0[2] + 0.05, round(std_0[2], decimal_round), ha='center', style='italic')
    axs[0].text(3, mean_0[3] + 0.05, round(std_0[3], decimal_round), ha='center', style='italic')

    axs[0].text(0, minimum_0[0] - 0.03, round(minimum_0[0], decimal_round), ha='center')
    # axs[0].text(1, minimum_0[1] - 0.03, round(minimum_0[1], decimal_round), ha='center')
    # axs[0].text(2, minimum_0[2] - 0.03, round(minimum_0[2], decimal_round), ha='center')
    # axs[0].text(3, minimum_0[3] - 0.03, round(minimum_0[3], decimal_round), ha='center')

    for i in range(0, n_labels):
        if maximum_0[i] > 0.15:
            axs[0].text(i, maximum_0[i] + 0.01, round(maximum_0[i], decimal_round), ha='center')

    for p in axs[0].patches:
        w = p.get_width()  # get width of bar
        axs[0].hlines(minimum_0[0], -w / 2, w / 2, colors='r')
        axs[0].hlines(minimum_0[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='r')
        axs[0].hlines(minimum_0[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='r')
        axs[0].hlines(minimum_0[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='r')

        axs[0].hlines(maximum_0[0], -w / 2, w / 2, colors='g')
        axs[0].hlines(maximum_0[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        axs[0].hlines(maximum_0[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')
        axs[0].hlines(maximum_0[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='g')

    axs[1].bar(labels, mean_1)
    # axs[1].set_title("predicted PUSHES")
    axs[1].set_title("predicted PUSHES (" + str(len(push_matrix)) + ")")

    axs[1].text(0, mean_1[0] + 0.1, round(mean_1[0], decimal_round), ha='center', fontweight='bold')
    axs[1].text(1, mean_1[1] - 0.05, round(mean_1[1], decimal_round), ha='center', fontweight='bold')
    axs[1].text(2, mean_1[2] + 0.1, round(mean_1[2], decimal_round), ha='center', fontweight='bold')
    axs[1].text(3, mean_1[3] + 0.1, round(mean_1[3], decimal_round), ha='center', fontweight='bold')

    axs[1].text(0, mean_1[0] + 0.05, round(std_1[0], decimal_round), ha='center', style='italic')
    axs[1].text(1, mean_1[1] - 0.1, round(std_1[1], decimal_round), ha='center', style='italic')
    axs[1].text(2, mean_1[2] + 0.05, round(std_1[2], decimal_round), ha='center', style='italic')
    axs[1].text(3, mean_1[3] + 0.05, round(std_1[3], decimal_round), ha='center', style='italic')

    # axs[1].text(0, minimum_1[0] - 0.03, round(minimum_1[0], decimal_round), ha='center')
    axs[1].text(1, minimum_1[1] - 0.03, round(minimum_1[1], decimal_round), ha='center')
    # axs[1].text(2, minimum_1[2] - 0.03, round(minimum_1[2], decimal_round), ha='center')
    # axs[1].text(3, minimum_1[3] - 0.03, round(minimum_1[3], decimal_round), ha='center')

    for i in range(0, n_labels):
        if maximum_1[i] > 0.15:
            axs[1].text(i, maximum_1[i] + 0.01, round(maximum_1[i], decimal_round), ha='center')

    for p in axs[1].patches:
        w = p.get_width()  # get width of bar
        axs[1].hlines(minimum_1[0], -w / 2, w / 2, colors='r')
        axs[1].hlines(minimum_1[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='r')
        axs[1].hlines(minimum_1[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='r')
        axs[1].hlines(minimum_1[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='r')

        axs[1].hlines(maximum_1[0], -w / 2, w / 2, colors='g')
        axs[1].hlines(maximum_1[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        axs[1].hlines(maximum_1[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')
        axs[1].hlines(maximum_1[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='g')

    axs[2].bar(labels, mean_2)
    # axs[2].set_title("predicted SHAKES")
    axs[2].set_title("predicted SHAKES (" + str(len(shake_matrix)) + ")")

    axs[2].text(0, mean_2[0] + 0.1, round(mean_2[0], decimal_round), ha='center', fontweight='bold')
    axs[2].text(1, mean_2[1] + 0.1, round(mean_2[1], decimal_round), ha='center', fontweight='bold')
    axs[2].text(2, mean_2[2] - 0.05, round(mean_2[2], decimal_round), ha='center', fontweight='bold')
    axs[2].text(3, mean_2[3] + 0.1, round(mean_2[3], decimal_round), ha='center', fontweight='bold')

    axs[2].text(0, mean_2[0] + 0.05, round(std_2[0], decimal_round), ha='center', style='italic')
    axs[2].text(1, mean_2[1] + 0.05, round(std_2[1], decimal_round), ha='center', style='italic')
    axs[2].text(2, mean_2[2] - 0.1, round(std_2[2], decimal_round), ha='center', style='italic')
    axs[2].text(3, mean_2[3] + 0.05, round(std_2[3], decimal_round), ha='center', style='italic')

    # axs[2].text(0, minimum_2[0] - 0.03, round(minimum_2[0], decimal_round), ha='center')
    # axs[2].text(1, minimum_2[1] - 0.03, round(minimum_2[1], decimal_round), ha='center')
    axs[2].text(2, minimum_2[2] - 0.03, round(minimum_2[2], decimal_round), ha='center')
    # axs[2].text(3, minimum_2[3] - 0.03, round(minimum_2[3], decimal_round), ha='center')

    for i in range(0, n_labels):
        if maximum_2[i] > 0.15:
            axs[2].text(i, maximum_2[i] + 0.01, round(maximum_2[i], decimal_round), ha='center')

    for p in axs[2].patches:
        w = p.get_width()  # get width of bar
        axs[2].hlines(minimum_2[0], -w / 2, w / 2, colors='r')
        axs[2].hlines(minimum_2[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='r')
        axs[2].hlines(minimum_2[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='r')
        axs[2].hlines(minimum_2[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='r')

        axs[2].hlines(maximum_2[0], -w / 2, w / 2, colors='g')
        axs[2].hlines(maximum_2[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        axs[2].hlines(maximum_2[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')
        axs[2].hlines(maximum_2[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='g')

    if n_labels == 4:
        axs[3].bar(labels, mean_3)
        axs[3].set_title("predicted TWISTS (" + str(len(twist_matrix)) + ")")

        axs[3].text(0, mean_3[0] + 0.1, round(mean_3[0], decimal_round), ha='center', fontweight='bold')
        axs[3].text(1, mean_3[1] + 0.1, round(mean_3[1], decimal_round), ha='center', fontweight='bold')
        axs[3].text(2, mean_3[2] + 0.1, round(mean_3[2], decimal_round), ha='center', fontweight='bold')
        axs[3].text(3, mean_3[3] - 0.05, round(mean_3[3], decimal_round), ha='center', fontweight='bold')

        axs[3].text(0, mean_3[0] + 0.05, round(std_3[0], decimal_round), ha='center', style='italic')
        axs[3].text(1, mean_3[1] + 0.05, round(std_3[1], decimal_round), ha='center', style='italic')
        axs[3].text(2, mean_3[2] + 0.05, round(std_3[2], decimal_round), ha='center', style='italic')
        axs[3].text(3, mean_3[3] - 0.1, round(std_3[3], decimal_round), ha='center', style='italic')

        # axs[1].text(0, minimum_1[0] - 0.03, round(minimum_1[0], decimal_round), ha='center')
        # axs[1].text(1, minimum_1[1] - 0.03, round(minimum_1[1], decimal_round), ha='center')
        # axs[1].text(2, minimum_1[2] - 0.03, round(minimum_1[2], decimal_round), ha='center')
        axs[3].text(3, minimum_3[3] - 0.03, round(minimum_3[3], decimal_round), ha='center')

        for i in range(0, n_labels):
            if maximum_3[i] > 0.15:
                axs[3].text(i, maximum_3[i] + 0.01, round(maximum_3[i], decimal_round), ha='center')

        for p in axs[3].patches:
            w = p.get_width()  # get width of bar
            axs[3].hlines(minimum_3[0], -w / 2, w / 2, colors='r')
            axs[3].hlines(minimum_3[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='r')
            axs[3].hlines(minimum_3[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='r')
            axs[3].hlines(minimum_3[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='r')

            axs[3].hlines(maximum_3[0], -w / 2, w / 2, colors='g')
            axs[3].hlines(maximum_3[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
            axs[3].hlines(maximum_3[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')
            axs[3].hlines(maximum_3[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='g')

    fig.suptitle('Output Confidence')

    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confidence_histogram.png", bbox_inches='tight')

    predicted_values = np.asarray(predictions_list)

    cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confusion_matrix.png", bbox_inches='tight')

    total_right, pull_acc, push_acc, shake_acc, twist_acc = 0, 0, 0, 0, 0

    if n_labels == 3:
        total_pull = cm[0][0] + cm[1][0] + cm[2][0]
        pull_acc = cm[0][0] / total_pull

        total_push = cm[0][1] + cm[1][1] + cm[2][1]
        push_acc = cm[1][1] / total_push

        total_shake = cm[0][2] + cm[1][2] + cm[2][2]
        shake_acc = cm[2][2] / total_shake

    elif n_labels == 4:
        total_pull = cm[0][0] + cm[1][0] + cm[2][0] + cm[3][0]
        pull_acc = cm[0][0] / total_pull

        total_push = cm[0][1] + cm[1][1] + cm[2][1] + cm[3][1]
        push_acc = cm[1][1] / total_push

        total_shake = cm[0][2] + cm[1][2] + cm[2][2] + cm[3][2]
        shake_acc = cm[2][2] / total_shake

        total_twist = cm[0][3] + cm[1][3] + cm[2][3] + cm[3][3]
        twist_acc = cm[3][3] / total_twist

    total = sum(sum(cm))

    print("total")
    print(total)

    if n_labels == 3:
        total_right = cm[0][0] + cm[1][1] + cm[2][2]
        accs = [pull_acc, push_acc, shake_acc]
        columns = ('PULL', 'PUSH', 'SHAKE')
    elif n_labels == 4:
        total_right = cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]
        accs = [pull_acc, push_acc, shake_acc, twist_acc]
        columns = ('PULL', 'PUSH', 'SHAKE', 'TWIST')

    print("total_right")
    print(total_right)

    rows = ["accuracy"]

    print("\n")

    print(Fore.LIGHTBLUE_EX + "Confusion Matrix Accuracy" + Fore.RESET)
    print(tabulate([accs], headers=labels, tablefmt="fancy_grid"))
    print("\n")
    print(
        Fore.LIGHTBLUE_EX + "Total Accuracy: " + Fore.LIGHTYELLOW_EX + str(round(total_right / total, 5)) + Fore.RESET)
    print("\n")
