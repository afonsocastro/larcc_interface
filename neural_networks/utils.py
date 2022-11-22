#!/usr/bin/env python3

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from json import JSONEncoder
import statistics


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


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {iteration} out of {total} Training & Test | ({percent}% {suffix})', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def prediction_classification(cla, true_out, dec_pred, dictionary, pred):
    if true_out == cla and dec_pred == cla:
        dictionary["true_positive"] = np.append(dictionary["true_positive"], pred, axis=0)
    elif true_out != cla and dec_pred == cla:
        dictionary["false_positive"] = np.append(dictionary["false_positive"], pred, axis=0)
    elif true_out == cla and dec_pred != cla:
        dictionary["false_negative"] = np.append(dictionary["false_negative"], pred, axis=0)
    elif true_out != cla and dec_pred != cla:
        dictionary["true_negative"] = np.append(dictionary["true_negative"], pred, axis=0)


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
