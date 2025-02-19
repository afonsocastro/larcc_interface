#!/usr/bin/env python3

import json
from larcc_interface.neural_networks.utils import simple_metrics_calc
import numpy as np
import scipy.stats as stats
from larcc_interface.neural_networks.utils import NumpyArrayEncoder

if __name__ == '__main__':
    transformer_model = "v1_4"
    json_file = transformer_model + "/100_times_train_test_transformer_"+transformer_model+".json"

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, list) and data:
        list_metrics_100 = {"accuracy": [], "recall": [], "precision": [], "f1": []}
        for i in range(0, len(data)):
            simulation = data[i]
            pull = simulation["test"]["push"]
            push = simulation["test"]["pull"]
            shake = simulation["test"]["shake"]
            twist = simulation["test"]["twist"]

            metrics_pull = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
            metrics_push = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
            metrics_shake = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
            metrics_twist = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

            simple_metrics_calc(pull, metrics_pull)
            simple_metrics_calc(push, metrics_push)
            simple_metrics_calc(shake, metrics_shake)
            simple_metrics_calc(twist, metrics_twist)

            for m in ["accuracy", "recall", "precision", "f1"]:
                list_metrics_100[m].append((metrics_pull[m] + metrics_push[m] + metrics_shake[m] + metrics_twist[m]) / 4)

        # irq - Interquartile Range , cv - Coefficient of Variation , 95% Confidence Interval, std_dev - standard deviation
        statistical_metrics = {"accuracy": {"mean": 0, "std_dev": 0, "iqr": 0, "cv": 0, "95_confidence_interval": 0},
                               "recall": {"mean": 0, "std_dev": 0, "iqr": 0, "cv": 0, "95_confidence_interval": 0},
                               "precision": {"mean": 0, "std_dev": 0, "iqr": 0, "cv": 0, "95_confidence_interval": 0},
                               "f1": {"mean": 0, "std_dev": 0, "iqr": 0, "cv": 0, "95_confidence_interval": 0}}

        for each_metric in ["accuracy", "recall", "precision", "f1"]:
            values_list = list_metrics_100[each_metric]
            mean = np.mean(values_list)
            std_dev = np.std(values_list, ddof=1)  # Sample standard deviation
            iqr = stats.iqr(values_list)
            cv = std_dev / mean if mean != 0 else float('inf')
            ci = stats.t.interval(0.95, len(values_list) - 1, loc=mean, scale=std_dev / np.sqrt(len(values_list)))

            statistical_metrics[each_metric]["mean"] = round(mean, 5)
            statistical_metrics[each_metric]["std_dev"] = round(std_dev, 5)
            statistical_metrics[each_metric]["iqr"] = round(iqr, 5)
            statistical_metrics[each_metric]["cv"] = round(cv, 5)
            statistical_metrics[each_metric]["95_confidence_interval"] = [round(ci[0], 5), round(ci[1], 5)]

        with open("statistical_metrics_100_dataset1_" + transformer_model + ".json", "w") as write_file:
            json.dump(statistical_metrics, write_file, cls=NumpyArrayEncoder)
    else:
        print("List is empty or invalid")




