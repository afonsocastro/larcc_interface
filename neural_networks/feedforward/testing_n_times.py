#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
from config.definitions import ROOT_DIR
from itertools import product
import numpy as np
from larcc_classes.documentation.PDF import PDF
import statistics
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def plot_confusion_matrix_percentage(confusion_matrix, display_labels=None, cmap="viridis",
                                     xticks_rotation="horizontal", title="Confusion Matrix", decimals=.1):
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
        text_cm = format(cm[i, j], str(decimals)+"f")
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


def values_contabilization(origin_dict, dest_dict):
    for indice, val in enumerate(origin_dict):
        dest_dict["accumulated"][indice] += val
        dest_dict["number"][indice] += 1


def mean_calc(origin_dict, dest_list, ind):
    mean = origin_dict["accumulated"][ind] / origin_dict["number"][ind]
    dest_list.append(mean)


def group_classification(origin_dict, dest_dict):
    dest_dict["true_positive"].append(len(origin_dict["true_positive"]))
    dest_dict["false_positive"].append(len(origin_dict["false_positive"]))
    dest_dict["false_negative"].append(len(origin_dict["false_negative"]))
    dest_dict["true_negative"].append(len(origin_dict["true_negative"]))


def metrics_calc(origin, metrics_dest, number):

    tp = origin["true_positive"][number]
    fp = origin["false_positive"][number]
    fn = origin["false_negative"][number]
    tn = origin["true_negative"][number]

    metric_accuracy = (tp + tn) / (fp + fn + tp + tn)
    metric_recall = tp / (fn + tp)
    metric_precision = tp / (fp + tp)
    metric_f1 = 2 * (metric_precision * metric_recall) / (metric_precision + metric_recall)

    metrics_dest["accuracy"].append(metric_accuracy)
    metrics_dest["recall"].append(metric_recall)
    metrics_dest["precision"].append(metric_precision)
    metrics_dest["f1"].append(metric_f1)


def filling_metrics_table(pull_metrics, push_metrics, shake_metrics, twist_metrics):
    data = [
        ["", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1", ],
        ["PULL", str(round(statistics.mean(pull_metrics["accuracy"]), 4)),
         str(round(statistics.mean(pull_metrics["precision"]), 4)),
         str(round(statistics.mean(pull_metrics["recall"]), 4)), str(round(statistics.mean(pull_metrics["f1"]), 4)), ],
        ["PUSH", str(round(statistics.mean(push_metrics["accuracy"]), 4)),
         str(round(statistics.mean(push_metrics["precision"]), 4)),
         str(round(statistics.mean(push_metrics["recall"]), 4)), str(round(statistics.mean(push_metrics["f1"]), 4)), ],
        ["SHAKE", str(round(statistics.mean(shake_metrics["accuracy"]), 4)),
         str(round(statistics.mean(shake_metrics["precision"]), 4)),
         str(round(statistics.mean(shake_metrics["recall"]), 4)), str(round(statistics.mean(shake_metrics["f1"]), 4)), ],
        ["TWIST", str(round(statistics.mean(twist_metrics["accuracy"]), 4)),
         str(round(statistics.mean(twist_metrics["precision"]), 4)),
         str(round(statistics.mean(twist_metrics["recall"]), 4)), str(round(statistics.mean(twist_metrics["f1"]), 4)), ],
    ]

    return data


def filling_metrics_table_n(pull_metrics, push_metrics, shake_metrics, twist_metrics, n):
    data = [
        ["test " + str(n), "Accuracy", "Precision", "Recall", "F1", ],
        ["PULL", str(round(pull_metrics["accuracy"][n], 4)), str(round(pull_metrics["precision"][n], 4)),
         str(round(pull_metrics["recall"][n], 4)), str(round(pull_metrics["f1"][n], 4)), ],
        ["PUSH", str(round(push_metrics["accuracy"][n], 4)), str(round(push_metrics["precision"][n], 4)),
         str(round(push_metrics["recall"][n], 4)), str(round(push_metrics["f1"][n], 4)), ],
        ["SHAKE", str(round(shake_metrics["accuracy"][n], 4)), str(round(shake_metrics["precision"][n], 4)),
         str(round(shake_metrics["recall"][n], 4)), str(round(shake_metrics["f1"][n], 4)), ],
        ["TWIST", str(round(twist_metrics["accuracy"][n], 4)), str(round(twist_metrics["precision"][n], 4)),
         str(round(twist_metrics["recall"][n], 4)), str(round(twist_metrics["f1"][n], 4)), ],

    ]

    return data


def filling_table(dict, header):
    data = [
        [header, "Mean", "Std Dev", "Max", "Min", ],
        ["True Positive", str(round(statistics.mean(dict["true_positive"]), 2)), str(round(statistics.stdev(dict["true_positive"]), 2)),
         str(max(dict["true_positive"])), str(min(dict["true_positive"])), ],
        ["False Positive", str(round(statistics.mean(dict["false_positive"]), 2)), str(round(statistics.stdev(dict["false_positive"]), 2)),
         str(max(dict["false_positive"])), str(min(dict["false_positive"])), ],
        ["False Negative", str(round(statistics.mean(dict["false_negative"]), 2)), str(round(statistics.stdev(dict["false_negative"]), 2)),
         str(max(dict["false_negative"])), str(min(dict["false_negative"])), ],
        # ["True Negative", str(round(statistics.mean(dict["true_negative"]), 2)), str(round(statistics.stdev(dict["true_negative"]), 2)),
        #  str(max(dict["true_negative"])), str(min(dict["true_negative"])), ],
    ]

    return data


if __name__ == '__main__':

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    output_neurons = 4

    model_config = json.load(open('model_config_optimized_' + str(output_neurons) + '_outputs.json'))
    max_epochs = model_config["epochs"]

    # # test_dict = {"cm_true": cm_true, "pull": pull, "push": push, "shake": shake, "twist": twist}
    # # training_test_dict = {"training": fit_history.history, "test": test_dict}
    # # training_test_list.append(training_test_dict)

    with open("training_testing_n_times/training_testing_n_times.json", "r") as read_file:
        training_test_list = json.load(read_file)

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

    pdf.output('metrics_table.pdf')

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

    pdf.output('table_class.pdf')

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

    plt.show()

    training_curves_mean_dict = {"loss": mean_loss, "accuracy": mean_accuracy, "val_loss": mean_val_loss,
                                 "val_accuracy": mean_val_accuracy}

    with open("training_testing_n_times/training_curves_mean.json", "w") as write_file:
        json.dump(training_curves_mean_dict, write_file, cls=NumpyArrayEncoder)

    # plt.savefig(ROOT_DIR + "/neural_networks/feedforward/training_testing_n_times/training_curves_mean.png",
    #             bbox_inches='tight')

    # -------------------------------------------------------------------------------------------------------------
    # CONFUSION MATRIX-----------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    blues = plt.cm.Blues
    cm_mean = np.empty((n_labels, n_labels))
    for i in range(0, cm_cumulative_percentage.shape[0]):
        for j in range(0, cm_cumulative_percentage.shape[1]):
            cm_mean[i][j] = cm_cumulative_percentage[i][j] / n_times

    cm_mean = cm_mean*100

    plot_confusion_matrix_percentage(confusion_matrix=cm_mean, display_labels=labels, cmap=blues,
                                     title="Mean Percentage CM (%d times)" % n_times, decimals=.2)
    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/feedforward/training_testing_n_times/confusion_matrix_mean_percentage.png",
                bbox_inches='tight')

    plot_confusion_matrix_percentage(confusion_matrix=cm_cumulative, display_labels=labels, cmap=blues,
                                     title="Cumulative CM (%d times)" % n_times, decimals=.0)
    plt.savefig(ROOT_DIR + "/neural_networks/feedforward/training_testing_n_times/confusion_matrix_cumulative.png",
                bbox_inches='tight')
    # -------------------------------------------------------------------------------------------------------------
