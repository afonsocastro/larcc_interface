#!/usr/bin/env python3

from tensorflow import keras
from itertools import product
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from config.definitions import ROOT_DIR
from tabulate import tabulate
from colorama import Fore


def save_txt_file(matrix, action, output):
    if not output:
        mat = np.matrix(matrix)
        with open(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_" + action + ".txt", 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')

    if output:
        output_mat = np.matrix(matrix)
        with open(ROOT_DIR + "/neural_networks/feedforward/predicted_data/output_predicted_" + action + ".txt",
                  'wb') as output_f:
            for output_line in output_mat:
                np.savetxt(output_f, output_line, fmt='%.2f')


def plot_confusion_matrix_percentage(confusion_matrix, display_labels=None, cmap="viridis",
                                     xticks_rotation="horizontal", title="Confusion Matrix"):
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
        # text_cm = format(cm[i, j], ".1f") + " %"
        text_cm = format(cm[i, j], ".1f")
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


def confidence_plot(ax, matrix, title, dr, nl):
    ax.set_title(title)
    ax.bar(labels, 0)
    if len(matrix) != 0:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # minimum = np.min(matrix, axis=0)
        # maximum = np.max(matrix, axis=0)

        ax.bar(labels, mean)

        for i in range(0, nl):
            if mean[i] < 0.2:
                ax.text(i, mean[i] + 0.1, round(mean[i], dr), ha='center', fontweight='bold')
                ax.text(i, mean[i] + 0.05, round(std[i], dr), ha='center', style='italic')
            else:
                ax.text(i, mean[i] - 0.05, round(mean[i], dr), ha='center', fontweight='bold')
                ax.text(i, mean[i] - 0.1, round(std[i], dr), ha='center', style='italic')

        #     if maximum[i] > 0.15:
        #         ax.text(i, maximum[i] + 0.01, round(maximum[i], dr), ha='center')
        #     if minimum[i] > 0.15:
        #         ax.text(i, minimum[i] - 0.03, round(minimum[i], dr), ha='center')
        #
        # for p in ax.patches:
        #     w = p.get_width()  # get width of bar
        #     ax.hlines(minimum[0], -w / 2, w / 2, colors='r')
        #     ax.hlines(minimum[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='r')
        #     ax.hlines(minimum[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='r')
        #     ax.hlines(minimum[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='r')
        #
        #     ax.hlines(maximum[0], -w / 2, w / 2, colors='g')
        #     ax.hlines(maximum[1], 0.2 + w / 2, 0.2 + w / 2 + w, colors='g')
        #     ax.hlines(maximum[2], w / 2 + 0.4 + w, w / 2 + 0.4 + w + w, colors='g')
        #     ax.hlines(maximum[3], w / 2 + 0.6 + w + w, w / 2 + 0.6 + w + w + w, colors='g')


def prediction_classification(cla, true_out, dec_pred, dictionary, pred):
    if true_out == cla and dec_pred == cla:
        dictionary["true_positive"] = np.append(dictionary["true_positive"], pred, axis=0)
    elif true_out != cla and dec_pred == cla:
        dictionary["false_positive"] = np.append(dictionary["false_positive"], pred, axis=0)
    elif true_out == cla and dec_pred != cla:
        dictionary["false_negative"] = np.append(dictionary["false_negative"], pred, axis=0)
    elif true_out != cla and dec_pred != cla:
        dictionary["true_negative"] = np.append(dictionary["true_negative"], pred, axis=0)


if __name__ == '__main__':

    # labels = ['PULL', 'PUSH', 'SHAKE']
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    # test_data = np.load(ROOT_DIR + '/data_storage/data/processed_learning_data/Ru_learning_data_4.npy',
    #                     mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    # test_data = np.load(ROOT_DIR + '/data_storage/data/processed_learning_data/Maf_learning_data_5.npy',
    #                     mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    model = keras.models.load_model("myModel")

    predictions_list = []

    col = test_data.shape[1]
    predicted_pull_matx = np.empty((0, col))
    predicted_push_matx = np.empty((0, col))
    predicted_shake_matx = np.empty((0, col))
    predicted_twist_matx = np.empty((0, col))

    output_predicted_pull = np.empty((0, n_labels))
    output_predicted_push = np.empty((0, n_labels))
    output_predicted_shake = np.empty((0, n_labels))
    output_predicted_twist = np.empty((0, n_labels))

    pull = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
            "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    push = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
            "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    shake = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
             "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}
    twist = {"true_positive": np.empty((0, n_labels)), "false_positive": np.empty((0, n_labels)),
             "false_negative": np.empty((0, n_labels)), "true_negative": np.empty((0, n_labels))}

    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=2)

        predict_lst = []
        for label in range(0, n_labels):
            predict_lst.append(prediction[0][label])

        decoded_prediction = np.argmax(prediction)
        true = test_data[i, -1]

        prediction_classification(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull, pred=prediction)
        prediction_classification(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push, pred=prediction)
        prediction_classification(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake, pred=prediction)
        prediction_classification(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist, pred=prediction)

        if decoded_prediction == 0:
            predicted_pull_matx = np.append(predicted_pull_matx, [test_data[i, :]], axis=0)
            output_predicted_pull = np.append(output_predicted_pull, prediction, axis=0)

        elif decoded_prediction == 1:
            predicted_push_matx = np.append(predicted_push_matx, [test_data[i, :]], axis=0)
            output_predicted_push = np.append(output_predicted_push, prediction, axis=0)

        elif decoded_prediction == 2:
            predicted_shake_matx = np.append(predicted_shake_matx, [test_data[i, :]], axis=0)
            output_predicted_shake = np.append(output_predicted_shake, prediction, axis=0)

        elif decoded_prediction == 3:
            predicted_twist_matx = np.append(predicted_twist_matx, [test_data[i, :]], axis=0)
            output_predicted_twist = np.append(output_predicted_twist, prediction, axis=0)

        predictions_list.append(decoded_prediction)

    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_pull.npy", predicted_pull_matx)
    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_push.npy", predicted_push_matx)
    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_shake.npy", predicted_shake_matx)

    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/output_predicted_pull.npy", output_predicted_pull)
    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/output_predicted_push.npy", output_predicted_push)
    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/output_predicted_shake.npy", output_predicted_shake)

    save_txt_file(predicted_pull_matx, "pull", False)
    save_txt_file(predicted_push_matx, "push", False)
    save_txt_file(predicted_shake_matx, "shake", False)

    save_txt_file(output_predicted_pull, "pull", True)
    save_txt_file(output_predicted_push, "push", True)
    save_txt_file(output_predicted_shake, "shake", True)

    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_twist.npy", predicted_twist_matx)
    np.save(ROOT_DIR + "/neural_networks/feedforward/predicted_data/output_predicted_twist.npy", output_predicted_twist)
    save_txt_file(predicted_twist_matx, "twist", False)
    save_txt_file(output_predicted_twist, "twist", True)

    decimal_round = 3

    # PULL ---------------------------------------------------------------------- PULL ------------------------------
    fig_pull, axs_pull = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    confidence_plot(ax=axs_pull[0], matrix=pull["true_positive"],
                    title="True Positives (" + str(len(pull["true_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_pull[1], matrix=pull["false_positive"],
                    title="False Positives (" + str(len(pull["false_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_pull[2], matrix=pull["false_negative"],
                    title="False Negatives (" + str(len(pull["false_negative"])) + ")", dr=decimal_round, nl=n_labels)

    # confidence_plot(ax=axs_pull[2], matrix=pull["true_negative"],
    #                 title="True Negatives (" + str(len(pull["true_negative"])) + ")", dr=decimal_round, nl=n_labels)

    fig_pull.suptitle('PULL Output Confidence')
    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/pull_output_confidences.png", bbox_inches='tight')

    # PUSH ---------------------------------------------------------------------- PUSH ------------------------------
    fig_push, axs_push = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    confidence_plot(ax=axs_push[0], matrix=push["true_positive"],
                    title="True Positives (" + str(len(push["true_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_push[1], matrix=push["false_positive"],
                    title="False Positives (" + str(len(push["false_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_push[2], matrix=push["false_negative"],
                    title="False Negatives (" + str(len(push["false_negative"])) + ")", dr=decimal_round, nl=n_labels)

    # confidence_plot(ax=axs_push[2], matrix=push["true_negative"],
    #                 title="True Negatives (" + str(len(push["true_negative"])) + ")", dr=decimal_round, nl=n_labels)

    fig_push.suptitle('PUSH Output Confidence')
    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/push_output_confidences.png", bbox_inches='tight')

    # SHAKE ---------------------------------------------------------------------- SHAKE ------------------------------
    fig_shake, axs_shake = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    confidence_plot(ax=axs_shake[0], matrix=shake["true_positive"],
                    title="True Positives (" + str(len(shake["true_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_shake[1], matrix=shake["false_positive"],
                    title="False Positives (" + str(len(shake["false_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_shake[2], matrix=shake["false_negative"],
                    title="False Negatives (" + str(len(shake["false_negative"])) + ")", dr=decimal_round, nl=n_labels)

    # confidence_plot(ax=axs_shake[2], matrix=shake["true_negative"],
    #                 title="True Negatives (" + str(len(shake["true_negative"])) + ")", dr=decimal_round, nl=n_labels)

    fig_shake.suptitle('SHAKE Output Confidence')
    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/shake_output_confidences.png", bbox_inches='tight')

    # TWIST ---------------------------------------------------------------------- TWIST ------------------------------
    fig_twist, axs_twist = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    confidence_plot(ax=axs_twist[0], matrix=twist["true_positive"],
                    title="True Positives (" + str(len(twist["true_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_twist[1], matrix=twist["false_positive"],
                    title="False Positives (" + str(len(twist["false_positive"])) + ")", dr=decimal_round, nl=n_labels)

    confidence_plot(ax=axs_twist[2], matrix=twist["false_negative"],
                    title="False Negatives (" + str(len(twist["false_negative"])) + ")", dr=decimal_round, nl=n_labels)

    # confidence_plot(ax=axs_twist[2], matrix=twist["true_negative"],
    #                 title="True Negatives (" + str(len(twist["true_negative"])) + ")", dr=decimal_round, nl=n_labels)

    fig_twist.suptitle('TWIST Output Confidence')
    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/twist_output_confidences.png", bbox_inches='tight')

    # --------------------------------------------------------------------------------------------------------------

    predicted_values = np.asarray(predictions_list)

    cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)
    print("cm")
    print(cm)

    cm_true = cm / cm.astype(float).sum(axis=1)
    cm_predicted = cm / cm.astype(float).sum(axis=0)
    cm_true_percentage = cm_true * 100
    cm_predicted_percentage = cm_predicted * 100

    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/confusion_matrix.png", bbox_inches='tight')

    plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
                                     title="Confusion Matrix (%)")

    plt.show()
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/confusion_matrix_true.png", bbox_inches='tight')

    # plot_confusion_matrix_percentage(confusion_matrix=cm_predicted_percentage, display_labels=labels, cmap=blues,
    #                                  title="Predicted Percentage CM (%)")
    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/predicted_data/confusion_matrix_predicted.png", bbox_inches='tight')

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

    print(Fore.LIGHTBLUE_EX + "Confusion Matrix Accuracy" + Fore.RESET)
    print(tabulate([accs_true], headers=labels, tablefmt="fancy_grid"))
    print("\n")

    # print(Fore.LIGHTBLUE_EX + "Confusion Matrix Predicted Accuracy" + Fore.RESET)
    # print(tabulate([accs_predicted], headers=labels, tablefmt="fancy_grid"))
    # print("\n")
    print(
        Fore.LIGHTBLUE_EX + "Total Accuracy: " + Fore.LIGHTYELLOW_EX + str(round(total_right / total, 5)) + Fore.RESET)
    print("\n")
