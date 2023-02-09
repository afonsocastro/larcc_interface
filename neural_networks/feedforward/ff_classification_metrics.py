#!/usr/bin/env python3

from tensorflow import keras
import numpy as np

from config.definitions import ROOT_DIR
from larcc_classes.documentation.PDF import PDF
from neural_networks.utils import filling_metrics_table_n, filling_metrics_table, simple_metrics_calc, filling_table


def prediction_classification(cla, true_out, dec_pred, dictionary):
    if true_out == cla and dec_pred == cla:
        dictionary["true_positive"] += 1
    elif true_out != cla and dec_pred == cla:
        dictionary["false_positive"] += 1
    elif true_out == cla and dec_pred != cla:
        dictionary["false_negative"] += 1
    elif true_out != cla and dec_pred != cla:
        dictionary["true_negative"] += 1


if __name__ == '__main__':

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

    model = keras.models.load_model("myModel")

    metrics_per_user = []

    pull = {"true_positive": 0, "false_positive": 0,
            "false_negative": 0, "true_negative": 0}
    push = {"true_positive": 0, "false_positive": 0,
            "false_negative": 0, "true_negative": 0}
    shake = {"true_positive": 0, "false_positive": 0,
             "false_negative": 0, "true_negative": 0}
    twist = {"true_positive": 0, "false_positive": 0,
             "false_negative": 0, "true_negative": 0}

    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i + 1, :-1], verbose=2)

        decoded_prediction = np.argmax(prediction)
        true = test_data[i, -1]

        prediction_classification(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
        prediction_classification(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
        prediction_classification(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
        prediction_classification(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

    # -------------------------------------------------------------------------------------------------------------
    # METRICS-----------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    metrics_pull = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    metrics_push = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    metrics_shake = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    metrics_twist = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

    simple_metrics_calc(pull, metrics_pull)
    simple_metrics_calc(push, metrics_push)
    simple_metrics_calc(shake, metrics_shake)
    simple_metrics_calc(twist, metrics_twist)

    metrics = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

    for m in ["accuracy", "recall", "precision", "f1"]:
        metrics[m] = (metrics_pull[m] + metrics_push[m] + metrics_shake[m] + metrics_twist[m]) / 4

    data = [
        ["8 users", "Accuracy", "Precision", "Recall", "F1", ],
        ["", str(round(metrics["accuracy"], 4)), str(round(metrics["precision"], 4)), str(round(metrics["recall"], 4)),
         str(round(metrics["f1"], 4)), ]
    ]

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=data, title='Metrics final test users', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output('metrics_final_test_users.pdf')
