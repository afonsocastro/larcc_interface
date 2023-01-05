#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
from config.definitions import ROOT_DIR
import numpy as np
from larcc_classes.documentation.PDF import PDF
import statistics
from neural_networks.utils import NumpyArrayEncoder, plot_confusion_matrix_percentage, values_contabilization, \
    group_classification, mean_calc, filling_metrics_table_n, filling_metrics_table, metrics_calc, filling_table


if __name__ == '__main__':

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    max_epochs = 40

    with open(ROOT_DIR + "/neural_networks/recurrent/seq2seq_n_times/training_testing_n_times_seq2seq.json", "r") as rf:
        training_test_list = json.load(rf)

    n_times = len(training_test_list)

    print("n_times")
    print(n_times)

    for n, training_test_dict in enumerate(training_test_list):
        training_test_dict["test"]["cm_true"] = np.asarray(training_test_dict["test"]["cm_true"])
        training_test_dict["test"]["cm"] = np.asarray(training_test_dict["test"]["cm"])

    cm_cumulative_percentage = np.zeros((n_labels, n_labels))
    cm_cumulative = np.zeros((n_labels, n_labels))

    loss = {"accumulated": [0] * max_epochs, "number": [0] * max_epochs}
    val_loss = {"accumulated": [0] * max_epochs, "number": [0] * max_epochs}
    accuracy = {"accumulated": [0] * max_epochs, "number": [0] * max_epochs}
    val_accuracy = {"accumulated": [0] * max_epochs, "number": [0] * max_epochs}

    pull = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    push = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    shake = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}
    twist = {"true_positive": [], "false_positive": [], "false_negative": [], "true_negative": []}

    for n_test in range(0, len(training_test_list)):

        # -------------------------------------------------------------------------------------------------------------
        # TRAINING CURVES-----------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        d = training_test_list[n_test]["training"]
        values_contabilization(origin_dict=d["loss"], dest_dict=loss)
        values_contabilization(origin_dict=d["val_loss"], dest_dict=val_loss)
        values_contabilization(origin_dict=d["accuracy"], dest_dict=accuracy)
        values_contabilization(origin_dict=d["val_accuracy"], dest_dict=val_accuracy)

        # -------------------------------------------------------------------------------------------------------------
        # CONFUSION MATRIX-----------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        for i in range(0, training_test_list[n_test]["test"]["cm_true"].shape[0]):
            for j in range(0, training_test_list[n_test]["test"]["cm_true"].shape[1]):
                cm_cumulative_percentage[i][j] += training_test_list[n_test]["test"]["cm_true"][i][j]
                cm_cumulative[i][j] += training_test_list[n_test]["test"]["cm"][i][j]

        # -------------------------------------------------------------------------------------------------------------
        # OUTPUT CONFIDENCES-----------------------------------------------------------------------------------------
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

    data_metrics_n = filling_metrics_table_n(pull_metrics=metrics_pull, push_metrics=metrics_push,
                                             shake_metrics=metrics_shake, twist_metrics=metrics_twist, n=0)
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=data_metrics_n, title='Metrics for one test', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=data_metrics, title='Mean Metrics', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output(ROOT_DIR + "/neural_networks/recurrent/seq2seq_n_times/metrics_table.pdf")

    # -------------------------------------------------------------------------------------------------------------
    # OUTPUT CONFIDENCES-----------------------------------------------------------------------------------------
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
    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=data_pull, title='PULL', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=data_push, title='PUSH', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=data_shake, title='SHAKE', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=data_twist, title='TWIST', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output(ROOT_DIR + "/neural_networks/recurrent/seq2seq_n_times/table_class.pdf")
    # -------------------------------------------------------------------------------------------------------------
    # TRAINING CURVES-----------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    mean_loss = []
    mean_val_loss = []
    mean_accuracy = []
    mean_val_accuracy = []

    for j in range(0, max_epochs):
        mean_calc(origin_dict=loss, dest_list=mean_loss, ind=j)
        mean_calc(origin_dict=val_loss, dest_list=mean_val_loss, ind=j)
        mean_calc(origin_dict=accuracy, dest_list=mean_accuracy, ind=j)
        mean_calc(origin_dict=val_accuracy, dest_list=mean_val_accuracy, ind=j)

    contribution_list = [x / len(training_test_list) for x in loss["number"]]

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(mean_accuracy)
    plt.plot(mean_val_accuracy)
    # plt.plot(contribution_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.legend(['train', 'val', 'attendance'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(mean_loss)
    plt.plot(mean_val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # plt.show()

    plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq_n_times/training_curves_mean.png")

    # -------------------------------------------------------------------------------------------------------------
    # CONFUSION MATRIX-----------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    blues = plt.cm.Blues
    cm_mean = np.empty((n_labels, n_labels))
    for i in range(0, cm_cumulative_percentage.shape[0]):
        for j in range(0, cm_cumulative_percentage.shape[1]):
            cm_mean[i][j] = cm_cumulative_percentage[i][j] / n_times

    cm_mean = cm_mean*100
    title = "Confusion Matrix (%) - Mean (" + str(
        n_times) + " times) \n RECURRENT LSTM Seq-To-Seq"
    plot_confusion_matrix_percentage(confusion_matrix=cm_mean, display_labels=labels, cmap=blues,
                                     title=title, decimals=.2)

    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq_n_times/confusion_matrix_mean_percentage.png",
                bbox_inches='tight')

    title = "Confusion Matrix - Cumulative (" + str(n_times) + " times) \n RECURRENT LSTM Seq-To-Seq"

    plot_confusion_matrix_percentage(confusion_matrix=cm_cumulative, display_labels=labels, cmap=blues,
                                     title=title, decimals=.0)

    plt.savefig(ROOT_DIR + "/neural_networks/recurrent/seq2seq_n_times/confusion_matrix_cumulative.png",
                bbox_inches='tight')
    # -------------------------------------------------------------------------------------------------------------
