#!/usr/bin/env python3

import json
from larcc_interface.neural_networks.utils import simple_metrics_calc
import numpy as np
import scipy.stats as stats
from larcc_interface.neural_networks.utils import NumpyArrayEncoder
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == '__main__':
    labels = ["Accuracy", "Precision", "Recall", "F1 score"]

    with open("v1_0/statistical_metrics_100_dataset1_v1_0.json", 'r', encoding='utf-8') as file_v1_0:
        data_v1_0 = json.load(file_v1_0)
    with open("v1_1/statistical_metrics_100_dataset1_v1_1.json", 'r', encoding='utf-8') as file_v1_1:
        data_v1_1 = json.load(file_v1_1)
    with open("v1_2/statistical_metrics_100_dataset1_v1_2.json", 'r', encoding='utf-8') as file_v1_2:
        data_v1_2 = json.load(file_v1_2)
    with open("v1_3/statistical_metrics_100_dataset1_v1_3.json", 'r', encoding='utf-8') as file_v1_3:
        data_v1_3 = json.load(file_v1_3)
    with open("v1_4/statistical_metrics_100_dataset1_v1_4.json", 'r', encoding='utf-8') as file_v1_4:
        data_v1_4 = json.load(file_v1_4)
    with open("v1_5/statistical_metrics_100_dataset1_v1_5.json", 'r', encoding='utf-8') as file_v1_5:
        data_v1_5 = json.load(file_v1_5)

    v1_0_mean = [data_v1_0["accuracy"]["mean"], data_v1_0["precision"]["mean"], data_v1_0["recall"]["mean"],
                 data_v1_0["f1"]["mean"]]
    v1_0_std_dev = [data_v1_0["accuracy"]["std_dev"], data_v1_0["precision"]["std_dev"], data_v1_0["recall"]["std_dev"],
                    data_v1_0["f1"]["std_dev"]]

    v1_1_mean = [data_v1_1["accuracy"]["mean"], data_v1_1["precision"]["mean"], data_v1_1["recall"]["mean"],
            data_v1_1["f1"]["mean"]]
    v1_1_std_dev = [data_v1_1["accuracy"]["std_dev"], data_v1_1["precision"]["std_dev"], data_v1_1["recall"]["std_dev"],
                    data_v1_1["f1"]["std_dev"]]

    v1_2_mean = [data_v1_2["accuracy"]["mean"], data_v1_2["precision"]["mean"], data_v1_2["recall"]["mean"],
            data_v1_2["f1"]["mean"]]
    v1_2_std_dev = [data_v1_2["accuracy"]["std_dev"], data_v1_2["precision"]["std_dev"], data_v1_2["recall"]["std_dev"],
                    data_v1_2["f1"]["std_dev"]]

    v1_3_mean = [data_v1_3["accuracy"]["mean"], data_v1_3["precision"]["mean"], data_v1_3["recall"]["mean"],
            data_v1_3["f1"]["mean"]]
    v1_3_std_dev = [data_v1_3["accuracy"]["std_dev"], data_v1_3["precision"]["std_dev"], data_v1_3["recall"]["std_dev"],
                    data_v1_3["f1"]["std_dev"]]

    v1_4_mean = [data_v1_4["accuracy"]["mean"], data_v1_4["precision"]["mean"], data_v1_4["recall"]["mean"],
            data_v1_4["f1"]["mean"]]
    v1_4_std_dev = [data_v1_4["accuracy"]["std_dev"], data_v1_4["precision"]["std_dev"], data_v1_4["recall"]["std_dev"],
                    data_v1_4["f1"]["std_dev"]]

    v1_5_mean = [data_v1_5["accuracy"]["mean"], data_v1_5["precision"]["mean"], data_v1_5["recall"]["mean"],
                 data_v1_5["f1"]["mean"]]
    v1_5_std_dev = [data_v1_5["accuracy"]["std_dev"], data_v1_5["precision"]["std_dev"], data_v1_5["recall"]["std_dev"],
                    data_v1_5["f1"]["std_dev"]]


    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    alpha = 0.5
    # Create translucent bars for the main values and error bars
    bars_v1_0 = ax.bar(x - (5*width/2), v1_0_mean, width, label='v1.0', color="#8f9bff", edgecolor="white",
           linewidth=2)
    ax.errorbar(x - (5*width/2), v1_0_mean, yerr=v1_0_std_dev, fmt='none',
                color="#8f9bff", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)

    bars_v1_1 = ax.bar(x - (3*width/2), v1_1_mean, width, label='v1.1', color="#47cd4d", edgecolor="white",
           linewidth=2)  # Translucent bars
    ax.errorbar(x - (3*width/2), v1_1_mean, yerr=v1_1_std_dev, fmt='none',
                color="#47cd4d", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)

    bars_v1_2 = ax.bar(x - (width/2), v1_2_mean, width, label='v1.2', color="#fe8281", edgecolor="white",
           linewidth=2)
    ax.errorbar(x - (width/2), v1_2_mean, yerr=v1_2_std_dev, fmt='none',
                color="#fe8281", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)

    bars_v1_3 = ax.bar(x + (width/2), v1_3_mean, width, label='v1.3', color="#e6df07", edgecolor="white",
           linewidth=2)
    ax.errorbar(x + (width/2), v1_3_mean, yerr=v1_3_std_dev, fmt='none',
                color="#e6df07", ecolor="black", elinewidth=2, capsize=5, alpha= alpha)

    bars_v1_4 = ax.bar(x + (3*width/2), v1_4_mean, width, label='v1.4', color="#b46eff", edgecolor="white",
                       linewidth=2)
    ax.errorbar(x + (3*width/2), v1_4_mean, yerr=v1_4_std_dev, fmt='none',
                color="#b46eff", ecolor="black", elinewidth=2, capsize=5, alpha=alpha)

    bars_v1_5 = ax.bar(x + (5*width/2), v1_5_mean, width, label='v1.5', color="#cc9600", edgecolor="white",
                       linewidth=2)
    ax.errorbar(x + (5*width/2), v1_5_mean, yerr=v1_5_std_dev, fmt='none',
                color="#cc9600", ecolor="black", elinewidth=2, capsize=5, alpha=alpha)

    # Add the values on top of each main bar (not on error bars)
    for bars, values in zip([bars_v1_0, bars_v1_1, bars_v1_2, bars_v1_3, bars_v1_4, bars_v1_5],
                        [v1_0_mean, v1_1_mean, v1_2_mean, v1_3_mean, v1_4_mean, v1_5_mean]):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value}',
                    ha='center', va='bottom', fontsize=10, color='black')

    # Add custom x-axis tick labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Mean & Standard Deviation - Transformers v1 - 100 simulations (train + dataset1 test)", fontsize=14, fontweight='bold')
    # Show the plot
    fig.tight_layout()
    delta = 0.03
    plt.ylim([min(v1_0_mean + v1_1_mean + v1_2_mean + v1_3_mean + v1_4_mean + v1_5_mean) - delta, max(v1_0_mean + v1_1_mean + v1_2_mean + v1_4_mean + v1_5_mean) + delta])
    plt.show()