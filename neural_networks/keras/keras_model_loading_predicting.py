#!/usr/bin/env python3

from tensorflow import keras
from itertools import product
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


def plot_confusion_matrix_percentage(confusion_matrix, display_labels=None, cmap="viridis", xticks_rotation="horizontal", title="Confusion Matrix"):
        colorbar = True
        im_kw = None
        fig, ax = plt.subplots()
        cm = confusion_matrix
        n_classes = cm.shape[0]

        default_im_kw = dict(interpolation="nearest", cmap=cmap)
        im_kw = im_kw or {}
        im_kw = {**default_im_kw, **im_kw}

        im_ = ax.imshow(cm, **im_kw)
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text_cm = format(cm[i, j], ".1f") + " %"
            text_[i, j] = ax.text(
                j, i, text_cm, ha="center", va="center", color=color
            )

        if display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = display_labels
        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        fig.suptitle(title)
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


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

    pull = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)), "false_negative": np.empty((0, n_labels))}
    push = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)), "false_negative": np.empty((0, n_labels))}
    shake = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)), "false_negative": np.empty((0, n_labels))}
    twist = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)), "false_negative": np.empty((0, n_labels))}

    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=2)

        predict_lst = []
        for label in range(0, n_labels):
            predict_lst.append(prediction[0][label])

        decoded_prediction = np.argmax(prediction)
        true = test_data[i, -1]

        for c in range(0, n_labels):
            if true == c & decoded_prediction == c:
                # TRUE POSITIVE
                true_positive = np.append(np.empty((0, n_labels)), prediction, axis=0)
            elif true != c & decoded_prediction == c:
                # FALSE POSITIVE
                false_positive = np.append(np.empty((0, n_labels)), prediction, axis=0)
            elif true == c & decoded_prediction != c:
                # FALSE NEGATIVE
                false_negative = np.append(np.empty((0, n_labels)), prediction, axis=0)

            if c == 0:
                pull["true_positive"] = np.append(pull["true_positive"], true_positive, axis=0)
                pull["false_positive"] = np.append(pull["false_positive"], false_positive, axis=0)
                pull["false_negative"] = np.append(pull["false_negative"], false_negative, axis=0)
            elif c == 1:
                push["true_positive"] = np.append(push["true_positive"], true_positive, axis=0)
                push["false_positive"] = np.append(push["false_positive"], false_positive, axis=0)
                push["false_negative"] = np.append(push["false_negative"], false_negative, axis=0)
            elif c == 2:
                shake["true_positive"] = np.append(shake["true_positive"], true_positive, axis=0)
                shake["false_positive"] = np.append(shake["false_positive"], false_positive, axis=0)
                shake["false_negative"] = np.append(shake["false_negative"], false_negative, axis=0)
            elif c == 3:
                twist["true_positive"] = np.append(twist["true_positive"], true_positive, axis=0)
                twist["false_positive"] = np.append(twist["false_positive"], false_positive, axis=0)
                twist["false_negative"] = np.append(twist["false_negative"], false_negative, axis=0)

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
    print("cm")
    print(cm)

    cm_true = cm / cm.astype(np.float).sum(axis=1)
    cm_predicted = cm / cm.astype(np.float).sum(axis=0)
    cm_true_percentage = cm_true * 100
    cm_predicted_percentage = cm_predicted * 100

    print("cm_true")
    print(cm_true)
    print("cm_predicted")
    print(cm_predicted)

    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confusion_matrix.png", bbox_inches='tight')

    plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues, title="True Percentage CM")
    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confusion_matrix_true.png", bbox_inches='tight')

    plot_confusion_matrix_percentage(confusion_matrix=cm_predicted_percentage, display_labels=labels, cmap=blues, title="Predicted Percentage CM")
    plt.savefig(ROOT_DIR + "/neural_networks/keras/predicted_data/confusion_matrix_predicted.png", bbox_inches='tight')

    pull_acc_true = cm_true[0][0]
    pull_acc_predicted = cm_predicted[0][0]

    push_acc_true = cm_true[1][1]
    push_acc_predicted = cm_predicted[1][1]

    shake_acc_true = cm_true[2][2]
    shake_acc_predicted = cm_predicted[2][2]
    if n_labels == 4:
        twist_acc_true = cm_true[3][3]
        twist_acc_predicted = cm_predicted[3][3]

    total = sum(sum(cm))

    print("total")
    print(total)

    total_right = cm[0][0] + cm[1][1] + cm[2][2]
    if n_labels == 3:
        accs_true = [pull_acc_true, push_acc_true, shake_acc_true]
        accs_predicted = [pull_acc_predicted, push_acc_predicted, shake_acc_predicted]
        columns = ('PULL', 'PUSH', 'SHAKE')
    elif n_labels == 4:
        total_right = total_right + cm[3][3]
        accs_true = [pull_acc_true, push_acc_true, shake_acc_true, twist_acc_true]
        accs_predicted = [pull_acc_predicted, push_acc_predicted, shake_acc_predicted, twist_acc_predicted]
        columns = ('PULL', 'PUSH', 'SHAKE', 'TWIST')

    print("total_right")
    print(total_right)

    print("\n")

    print(Fore.LIGHTBLUE_EX + "Confusion Matrix True Accuracy" + Fore.RESET)
    print(tabulate([accs_true], headers=labels, tablefmt="fancy_grid"))
    print("\n")

    print(Fore.LIGHTBLUE_EX + "Confusion Matrix Predicted Accuracy" + Fore.RESET)
    print(tabulate([accs_predicted], headers=labels, tablefmt="fancy_grid"))
    print("\n")
    print(
        Fore.LIGHTBLUE_EX + "Total Accuracy: " + Fore.LIGHTYELLOW_EX + str(round(total_right / total, 5)) + Fore.RESET)
    print("\n")
