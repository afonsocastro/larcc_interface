#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
from config.definitions import ROOT_DIR
import numpy as np
import glob, os
import re, statistics
from larcc_classes.documentation.PDF import PDF
from neural_networks.utils import NumpyArrayEncoder, plot_confusion_matrix_percentage, values_contabilization, \
    group_classification, mean_calc, filling_metrics_table_n, filling_metrics_table, metrics_calc, filling_table


def create_grouped_bar_chart(labels, global_mean_metrics, title, ylabel):
    plt.close("all")  # this is the line to be added

    lstm = [float(global_mean_metrics[1][1]), float(global_mean_metrics[1][2]), float(global_mean_metrics[1][3]), float(global_mean_metrics[1][4])]
    gru = [float(global_mean_metrics[2][1]), float(global_mean_metrics[2][2]), float(global_mean_metrics[2][3]), float(global_mean_metrics[2][4])]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()

    rects_lstm = ax.bar(x - width/2, lstm, width, label='LSTM', color="#47cd4d", edgecolor="white", linewidth=2)
    rects_gru = ax.bar(x + width/2, gru, width, label='GRU', color="#8f9bff", edgecolor="white", linewidth=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects_lstm, padding=3)
    ax.bar_label(rects_gru, padding=3)

    fig.tight_layout()
    plt.ylim([min(lstm + gru) - 0.01, max(lstm + gru) + 0.01])
    plt.show()

    # plt.savefig(ROOT_DIR + "/neural_networks/convolutional/n_times_results/mean_results_" + str(
    #     ylabel) + ".png")


def create_global_grouped_bar_chart(accuracy, precision, recall, f1, title):
    # plt.close("all")  # this is the line to be added
    labels=["Accuracy", "Precision", "Recall", "F1 Score"]

    lstm = [statistics.mean([float(accuracy[1][1]), float(accuracy[1][2]), float(accuracy[1][3]), float(accuracy[1][4])]),
          statistics.mean(
              [float(precision[1][1]), float(precision[1][2]), float(precision[1][3]), float(precision[1][4])]),
          statistics.mean([float(recall[1][1]), float(recall[1][2]), float(recall[1][3]), float(recall[1][4])]),
          statistics.mean([float(f1[1][1]), float(f1[1][2]), float(f1[1][3]), float(f1[1][4])])]

    gru = [
        statistics.mean([float(accuracy[2][1]), float(accuracy[2][2]), float(accuracy[2][3]), float(accuracy[2][4])]),
        statistics.mean(
            [float(precision[2][1]), float(precision[2][2]), float(precision[2][3]), float(precision[2][4])]),
        statistics.mean([float(recall[2][1]), float(recall[2][2]), float(recall[2][3]), float(recall[2][4])]),
        statistics.mean([float(f1[2][1]), float(f1[2][2]), float(f1[2][3]), float(f1[2][4])])]

    lstm = [round(item, 4) for item in lstm]
    gru = [round(item, 4) for item in gru]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    # rects_cnn0 = ax.bar(x - (2 * width), cnn0, width, label='CNN 0', color="#8f9bff", edgecolor="white", linewidth=2)
    # rects_cnn1 = ax.bar(x - width, cnn1, width, label='CNN 1', color="#47cd4d", edgecolor="white", linewidth=2)
    rects_lstm = ax.bar(x - width/2, lstm, width, label='LSTM', color="#47cd4d", edgecolor="white", linewidth=2)
    rects_gru = ax.bar(x + width/2, gru, width, label='GRU', color="#8f9bff", edgecolor="white", linewidth=2)
    # rects_cnn3 = ax.bar(x + width, cnn3, width, label='CNN 3', color="#edac5a", edgecolor="white", linewidth=2)
    # rects_cnn4 = ax.bar(x + (2 * width), cnn4, width, label='CNN 4', color="#e466ff", edgecolor="white", linewidth=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects_lstm, padding=3)
    ax.bar_label(rects_gru, padding=3)

    fig.tight_layout()
    plt.ylim([min(lstm + gru) - 0.005, max(lstm + gru) + 0.005])
    plt.show()

    # plt.savefig(ROOT_DIR + "/neural_networks/convolutional/n_times_results/mean_GLOBAL_results.png")


if __name__ == '__main__':

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    total_metrics = []

    for infile in sorted(glob.glob('n_times_train_test/*.json')):
        print("Current File Being Processed is: " + infile)

        with open(infile, "r") as read_file:
            training_test_list = json.load(read_file)

        n_times = len(training_test_list)

        print("n_times")
        print(n_times)

        for n, training_test_dict in enumerate(training_test_list):
            training_test_dict["test"]["cm_true"] = np.asarray(training_test_dict["test"]["cm_true"])
            training_test_dict["test"]["cm"] = np.asarray(training_test_dict["test"]["cm"])

        cm_cumulative_percentage = np.zeros((n_labels, n_labels))
        cm_cumulative = np.zeros((n_labels, n_labels))

        pull = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
        push = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
        shake = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
        twist = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}

        for n_test in range(0, n_times):

            # # -------------------------------------------------------------------------------------------------------------
            # # TRAINING CURVES--------------------------------------------------------------------------------------
            # # -------------------------------------------------------------------------------------------------------------
            # d = training_test_list[n_test]["training"]
            # epochs = len(d["training"]["loss"])
            #
            # loss = {"accumulated": [0] * epochs, "number": [0] * epochs}
            # val_loss = {"accumulated": [0] * epochs, "number": [0] * epochs}
            # accuracy = {"accumulated": [0] * epochs, "number": [0] * epochs}
            # val_accuracy = {"accumulated": [0] * epochs, "number": [0] * epochs}
            # values_contabilization(origin_dict=d["loss"], dest_dict=loss)
            # values_contabilization(origin_dict=d["val_loss"], dest_dict=val_loss)
            # values_contabilization(origin_dict=d["accuracy"], dest_dict=accuracy)
            # values_contabilization(origin_dict=d["val_accuracy"], dest_dict=val_accuracy)

            # -------------------------------------------------------------------------------------------------------------
            # CONFUSION MATRIX-------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------------

            for i in range(0, training_test_list[n_test]["test"]["cm_true"].shape[0]):
                for j in range(0, training_test_list[n_test]["test"]["cm_true"].shape[1]):
                    cm_cumulative_percentage[i][j] += training_test_list[n_test]["test"]["cm_true"][i][j]
                    cm_cumulative[i][j] += training_test_list[n_test]["test"]["cm"][i][j]

            # -------------------------------------------------------------------------------------------------------------
            # OUTPUT CONFIDENCES-----------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------------
            dt = training_test_list[n_test]["test"]
            group_classification(origin_dict=dt["pull"], dest_dict=pull)
            group_classification(origin_dict=dt["push"], dest_dict=push)
            group_classification(origin_dict=dt["shake"], dest_dict=shake)
            group_classification(origin_dict=dt["twist"], dest_dict=twist)

        # -------------------------------------------------------------------------------------------------------------
        # METRICS-----------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        metrics_pull = {"accuracy": [], "recall": [], "precision": [], "f1": []}
        metrics_push = {"accuracy": [], "recall": [], "precision": [], "f1": []}
        metrics_shake = {"accuracy": [], "recall": [], "precision": [], "f1": []}
        metrics_twist = {"accuracy": [], "recall": [], "precision": [], "f1": []}

        for l in range(0, len(pull["true_positive"])):
            metrics_calc(pull, metrics_pull, l)
            metrics_calc(push, metrics_push, l)
            metrics_calc(shake, metrics_shake, l)
            metrics_calc(twist, metrics_twist, l)

        data_metrics = filling_metrics_table(pull_metrics=metrics_pull, push_metrics=metrics_push,
                                             shake_metrics=metrics_shake, twist_metrics=metrics_twist)

        # metrics_per_nn.append({"model": n_model, "metrics": data_metrics})
        total_metrics.append(data_metrics)
        data_metrics_n = filling_metrics_table_n(pull_metrics=metrics_pull, push_metrics=metrics_push,
                                                 shake_metrics=metrics_shake, twist_metrics=metrics_twist, n=0)
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Times", size=10)

        pdf.create_table(table_data=data_metrics_n, title='Metrics for one test', cell_width='uneven', x_start=25)
        pdf.ln()

        pdf.create_table(table_data=data_metrics, title='Mean Metrics', cell_width='uneven', x_start=25)
        pdf.ln()

        pdf.output('n_times_results/LSTM_metrics_table.pdf')

        # -------------------------------------------------------------------------------------------------------------
        # OUTPUT CONFIDENCES---------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        total_pull = sum(training_test_list[0]["test"]["cm"][0])
        total_push = sum(training_test_list[0]["test"]["cm"][1])
        total_shake = sum(training_test_list[0]["test"]["cm"][2])
        total_twist = sum(training_test_list[0]["test"]["cm"][3])

        data_pull = filling_table(dict=pull, header="%d" % total_pull)
        data_push = filling_table(dict=push, header="%d" % total_push)
        data_shake = filling_table(dict=shake, header="%d" % total_shake)
        data_twist = filling_table(dict=twist, header="%d" % total_twist)

        pdf = PDF()
        pdf.add_page()
        pdf.set_title("LSTM recurrentNN Output results \n study of %d times" % n_times)

        pdf.set_font("Times", size=10)

        pdf.create_table(table_data=data_pull, title='PULL', cell_width='uneven', x_start=25)
        pdf.ln()

        pdf.create_table(table_data=data_push, title='PUSH', cell_width='uneven', x_start=25)
        pdf.ln()

        pdf.create_table(table_data=data_shake, title='SHAKE', cell_width='uneven', x_start=25)
        pdf.ln()

        pdf.create_table(table_data=data_twist, title='TWIST', cell_width='uneven', x_start=25)
        pdf.ln()

        pdf.output('n_times_results/LSTM_output_table.pdf')

        # -------------------------------------------------------------------------------------------------------------
        # TRAINING CURVES-----------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        mean_loss = []
        mean_val_loss = []
        mean_accuracy = []
        mean_val_accuracy = []

        # for j in range(0, epochs):
        #     mean_calc(origin_dict=loss, dest_list=mean_loss, ind=j)
        #     mean_calc(origin_dict=val_loss, dest_list=mean_val_loss, ind=j)
        #     mean_calc(origin_dict=accuracy, dest_list=mean_accuracy, ind=j)
        #     mean_calc(origin_dict=val_accuracy, dest_list=mean_val_accuracy, ind=j)

        # contribution_list = [x / len(training_test_list) for x in loss["number"]]

        # fig = plt.figure()
        #
        # plt.subplot(1, 2, 1)
        # plt.plot(mean_accuracy)
        # plt.plot(mean_val_accuracy)
        # plt.plot(contribution_list)
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # # plt.legend(['train', 'val'], loc='upper left')
        # plt.legend(['train', 'val', 'attendance'], loc='upper left')
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(mean_loss)
        # plt.plot(mean_val_loss)
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        #
        # plt.show()

        # training curves data for tickz
        # training_curves_mean_dict = {"loss": mean_loss, "accuracy": mean_accuracy, "val_loss": mean_val_loss,
        #                              "val_accuracy": mean_val_accuracy}

        # with open("training_testing_n_times/training_curves_mean.json", "w") as write_file:
        #     json.dump(training_curves_mean_dict, write_file, cls=NumpyArrayEncoder)

        # plt.savefig(ROOT_DIR + "/neural_networks/convolutional/n_times_results/training_curves_mean" + str(
        #     n_model) + ".png")
        # bbox_inches = 'tight'

        # -------------------------------------------------------------------------------------------------------------
        # CONFUSION MATRIX-----------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        blues = plt.cm.Blues
        cm_mean = np.empty((n_labels, n_labels))
        for i in range(0, cm_cumulative_percentage.shape[0]):
            for j in range(0, cm_cumulative_percentage.shape[1]):
                cm_mean[i][j] = cm_cumulative_percentage[i][j] / n_times

        cm_mean = cm_mean * 100
        title = "Confusion Matrix (%) - Mean (" + str(
            n_times) + " times) \n LSTM RNN"
        plot_confusion_matrix_percentage(confusion_matrix=cm_mean, display_labels=labels, cmap=blues,
                                         title=title, decimals=.2)

        plt.show()
        # plt.savefig(
        #     ROOT_DIR + "/neural_networks/convolutional/n_times_results/confusion_matrix_mean_percentage_" + str(
        #         n_model) + ".png", bbox_inches='tight')

        title = "Confusion Matrix - Cumulative (" + str(n_times) + " times) \n LSTM RNN"

        plot_confusion_matrix_percentage(confusion_matrix=cm_cumulative, display_labels=labels, cmap=blues,
                                         title=title, decimals=.0)

        # plt.savefig(ROOT_DIR + "/neural_networks/convolutional/n_times_results/confusion_matrix_cumulative_" + str(
        #     n_model) + ".png", bbox_inches='tight')
        # -------------------------------------------------------------------------------------------------------------

    # For every file (Global evaluation)
    global_mean_metrics_accuracy = [[str(n_times) + " times", "PULL", "PUSH", "SHAKE", "TWIST", ], ]
    global_mean_metrics_precision = [[str(n_times) + " times", "PULL", "PUSH", "SHAKE", "TWIST", ], ]
    global_mean_metrics_recall = [[str(n_times) + " times", "PULL", "PUSH", "SHAKE", "TWIST", ], ]
    global_mean_metrics_f1 = [[str(n_times) + " times", "PULL", "PUSH", "SHAKE", "TWIST", ], ]

    global_mean_metrics_accuracy.append(
        ["LSTM", total_metrics[0][1][1], total_metrics[0][2][1], total_metrics[0][3][1], total_metrics[0][4][1], ])
    global_mean_metrics_accuracy.append(
        ["GRU", total_metrics[1][1][1], total_metrics[1][2][1], total_metrics[1][3][1], total_metrics[1][4][1], ])

    global_mean_metrics_precision.append(
        ["LSTM", total_metrics[0][1][2], total_metrics[0][2][2], total_metrics[0][3][2], total_metrics[0][4][2], ])
    global_mean_metrics_precision.append(
        ["GRU", total_metrics[1][1][2], total_metrics[1][2][2], total_metrics[1][3][2], total_metrics[1][4][2], ])

    global_mean_metrics_recall.append(
        ["LSTM", total_metrics[0][1][3], total_metrics[0][2][3], total_metrics[0][3][3], total_metrics[0][4][3], ])
    global_mean_metrics_recall.append(
        ["GRU", total_metrics[1][1][3], total_metrics[1][2][3], total_metrics[1][3][3], total_metrics[1][4][3], ])

    global_mean_metrics_f1.append(
        ["LSTM", total_metrics[0][1][4], total_metrics[0][2][4], total_metrics[0][3][4], total_metrics[0][4][4], ])
    global_mean_metrics_f1.append(
        ["GRU", total_metrics[1][1][4], total_metrics[1][2][4], total_metrics[1][3][4], total_metrics[1][4][4], ])

    pdf = PDF()
    pdf.add_page()

    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=global_mean_metrics_accuracy, title='Mean Accuracy', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=global_mean_metrics_precision, title='Mean Precision', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=global_mean_metrics_recall, title='Mean Recall', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=global_mean_metrics_f1, title='Mean F1 Score', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output('n_times_results/evaluation_LSTM.pdf')

    create_grouped_bar_chart(labels, global_mean_metrics_accuracy, title="Mean Accuracy (100 times)", ylabel="accuracy")
    create_grouped_bar_chart(labels, global_mean_metrics_precision, title="Mean Precision (100 times)", ylabel="precision")
    create_grouped_bar_chart(labels, global_mean_metrics_recall, title="Mean Recall (100 times)", ylabel="recall")
    create_grouped_bar_chart(labels, global_mean_metrics_f1, title="Mean F1 score (100 times)", ylabel="f1-score")

    create_global_grouped_bar_chart(global_mean_metrics_accuracy, global_mean_metrics_precision,
                                    global_mean_metrics_recall, global_mean_metrics_f1,
                                    title="Mean Score per metric per RNN\n(4 primitives - 100 times)")

