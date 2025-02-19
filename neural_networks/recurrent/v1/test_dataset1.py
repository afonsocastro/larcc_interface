#!/usr/bin/env python3

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from larcc_interface.larcc_classes.documentation.PDF import PDF
from larcc_interface.neural_networks.utils import plot_confusion_matrix_percentage, prediction_classification, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay
from larcc_interface.config.definitions import ROOT_DIR
import numpy as np


if __name__ == '__main__':
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    input_nn = 20

    test_data_1 = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_test_data_20ms.npy")

    x_test = test_data_1[:, :-1]
    y_test = test_data_1[:, -1]
    x_test = np.reshape(x_test, (test_data_1.shape[0], input_nn, 13, 1))

    x_test = x_test[:, :, 1:, :]

    version="v1_4"
    model_name = "lstm_"+version
    model = keras.models.load_model(version+"/lstm_"+version+".keras")


    predictions_list = []

    pull = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    push = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    shake = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    twist = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

    for i in range(0, len(test_data_1)):
        # prediction = transformer_v1_0.predict(x=x_test[i:i + 1, :, :, :], verbose=0)
        prediction = model.predict(x=x_test[i:i + 1, :, :, :], verbose=0)

        # Reverse to_categorical from keras utils
        decoded_prediction = np.argmax(prediction, axis=1, out=None)

        # true = test_data[i, -1]
        true = y_test[i]

        prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
        prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
        prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
        prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

        predictions_list.append(decoded_prediction)

    predicted_values = np.asarray(predictions_list)

    cm = confusion_matrix(y_true=y_test, y_pred=predicted_values)

    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    plt.show()

    cm_true = cm / cm.astype(float).sum(axis=1)
    cm_true_percentage = cm_true * 100
    plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
                                     title="Confusion Matrix (%) - " + model_name)
    plt.show()

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
        ["", "Accuracy", "Precision", "Recall", "F1", ],
        ["", str(round(metrics["accuracy"], 4)), str(round(metrics["precision"], 4)), str(round(metrics["recall"], 4)),
         str(round(metrics["f1"], 4)), ]
    ]

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=data, title='Metrics Dataset 1', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output(model_name + '.pdf')