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

    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    n_labels = len(labels)

    plt.close("all")  # this is the line to be added

    t_ff = [0.9767, 0.9554, 0.9537, 0.9533]

    t_cnn0 = [0.9826, 0.9655, 0.9651, 0.9649]
    t_cnn1 = [0.9823, 0.9657, 0.9649, 0.9644]
    t_cnn2 = [0.9834, 0.9677, 0.9671, 0.9667]
    t_cnn3 = [0.9848, 0.9703, 0.9698, 0.9695]
    t_cnn4 = [0.9821, 0.9647, 0.9642, 0.964]

    t_cnn_1s = [0.9708, 0.9427, 0.9411, 0.9409]
    t_cnn_2s = [0.9777, 0.9559, 0.9552, 0.955]
    t_cnn_3s = [0.9766, 0.9539, 0.9532, 0.9528]
    t_cnn_4s = [0.9797, 0.96, 0.9594, 0.9591]

    # t_rr_seq2label = [0.9674, 0.8859, 1, 0.9395]  #PULL
    # t_rr_seq2label = [0.9885, 1, 0.9535, 0.9762]  #PUSH
    # t_rr_seq2label = [0.9272, 0.887, 0.8031, 0.843]  #SHAKE
    # t_rr_seq2label = [0.9406, 0.8815, 0.8881, 0.8848]  #TWIST

    t_rr_seq2label = [0.9559, 0.9136, 0.911175, 0.910875]

    x = np.arange(len(labels))  # the label locations
    width = 0.07  # the width of the bars

    fig, ax = plt.subplots()
    rects_tff = ax.bar(x - (5 * width), t_ff, width, label='FeedForward', color="#7f8c8d", edgecolor="white", linewidth=2)
    rects_tcnn0 = ax.bar(x - (4 * width), t_cnn0, width, label='cnn 0', color="#47cd4d", edgecolor="white", linewidth=2)
    rects_tcnn1 = ax.bar(x - (3 * width), t_cnn1, width, label='cnn 1', color="#fe8281", edgecolor="white", linewidth=2)
    rects_tcnn2 = ax.bar(x - (2 * width), t_cnn2, width, label='cnn 2', color="#8f9bff", edgecolor="white", linewidth=2)
    rects_tcnn3 = ax.bar(x - width, t_cnn3, width, label='cnn 3', color="#edac5a", edgecolor="white", linewidth=2)
    rects_tcnn4 = ax.bar(x, t_cnn4, width, label='cnn 4', color="#d2b4de", edgecolor="white", linewidth=2)
    rects_tcnn_1s = ax.bar(x + width, t_cnn_1s, width, label='0.1 s', color="#1e8449", edgecolor="white", linewidth=2)
    rects_tcnn_2s = ax.bar(x + (2 * width), t_cnn_2s, width, label='0.2 s', color="#1f618d", edgecolor="white", linewidth=2)
    rects_tcnn_3s = ax.bar(x + (3 * width), t_cnn_3s, width, label='0.3 s', color="#a93226", edgecolor="white", linewidth=2)
    rects_tcnn_4s = ax.bar(x + (4 * width), t_cnn_4s, width, label='0.4 s', color="#d4ac0d", edgecolor="white", linewidth=2)
    rects_trr = ax.bar(x + (5 * width), t_rr_seq2label, width, label='Seq2Label (rnn)', color="#17202a", edgecolor="white", linewidth=2)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("ylabel")

    ax.set_title("title")
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects_tff, padding=3)
    ax.bar_label(rects_tcnn0, padding=3)
    ax.bar_label(rects_tcnn1, padding=3)
    ax.bar_label(rects_tcnn2, padding=3)
    ax.bar_label(rects_tcnn3, padding=3)
    ax.bar_label(rects_tcnn4, padding=3)
    ax.bar_label(rects_tcnn_1s, padding=3)
    ax.bar_label(rects_tcnn_2s, padding=3)
    ax.bar_label(rects_tcnn_3s, padding=3)
    ax.bar_label(rects_tcnn_4s, padding=3)
    ax.bar_label(rects_trr, padding=3)

    fig.tight_layout()
    plt.ylim([0.9, 0.99])
    plt.show()
