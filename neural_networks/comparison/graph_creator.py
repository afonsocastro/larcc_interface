#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
from config.definitions import ROOT_DIR
import numpy as np
import glob, os
from larcc_classes.documentation.PDF import PDF
import statistics
from neural_networks.utils import NumpyArrayEncoder, plot_confusion_matrix_percentage, values_contabilization, \
    group_classification, mean_calc, filling_metrics_table_n, filling_metrics_table, metrics_calc, filling_table


if __name__ == '__main__':

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    plt.close("all")  # this is the line to be added

    t1 = [12, 14, 9, 8]
    t2 = [11, 10, 7, 12]
    t3 = [15, float(global_mean_metrics[3][2]), float(global_mean_metrics[3][3]),
          float(global_mean_metrics[3][4])]
    t4 = [float(global_mean_metrics[4][1]), float(global_mean_metrics[4][2]), float(global_mean_metrics[4][3]),
          float(global_mean_metrics[4][4])]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects_t1 = ax.bar(x - (3 * width / 2), t1, width, label='0.1 s', color="#8f9bff", edgecolor="white", linewidth=2)
    rects_t2 = ax.bar(x - width / 2, t2, width, label='0.2 s', color="#47cd4d", edgecolor="white", linewidth=2)
    rects_t3 = ax.bar(x + width / 2, t3, width, label='0.3 s', color="#fe8281", edgecolor="white", linewidth=2)
    rects_t4 = ax.bar(x + (3 * width / 2), t4, width, label='0.4 s', color="#edac5a", edgecolor="white", linewidth=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects_t1, padding=3)
    ax.bar_label(rects_t2, padding=3)
    ax.bar_label(rects_t3, padding=3)
    ax.bar_label(rects_t4, padding=3)

    fig.tight_layout()
    plt.ylim([min(t1 + t2 + t3 + t4) - 0.01, max(t1 + t2 + t3 + t4) + 0.01])
    plt.show()