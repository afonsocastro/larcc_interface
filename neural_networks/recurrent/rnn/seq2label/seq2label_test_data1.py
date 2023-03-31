#!/usr/bin/env python3
#
import keras
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from larcc_classes.documentation.PDF import PDF
from neural_networks.utils import plot_confusion_matrix_percentage, prediction_classification, simple_metrics_calc, \
    prediction_classification_absolute
from sklearn.metrics import ConfusionMatrixDisplay

from config.definitions import ROOT_DIR
from larcc_classes.data_storage.SortedDataForLearning import SortedDataForLearning
import numpy as np


if __name__ == '__main__':
    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150

    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    test_data = np.load(ROOT_DIR + "/data_storage/data1/global_normalized_test_data_20ms.npy")

    n_test = test_data.shape[0]
    x_test = np.reshape(test_data[:, :-1], (n_test, time_steps, 13))
    y_test = to_categorical(test_data[:, -1])

    x_test = x_test[:, :, 1:]

    print(x_test.shape)
    print(y_test.shape)

    seq2label_model = keras.models.load_model("seq2label_20ms")

    predictions_list = []

    pull = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    push = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    shake = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    twist = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

    for i in range(0, len(test_data)):
        x = np.reshape(x_test[i], (1, x_test[i].shape[0], x_test[i].shape[1]))
        prediction = seq2label_model.predict(x=x, verbose=0)

        # Reverse to_categorical from keras utils
        decoded_prediction = np.argmax(prediction, axis=1, out=None)

        true = test_data[i, -1]

        prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
        prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
        prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
        prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

        predictions_list.append(decoded_prediction)

    predicted_values = np.asarray(predictions_list)

    # Reverse to_categorical from keras utils
    # predicted_values = np.argmax(predicted_values, axis=1, out=None)

    true_values = test_data[:, -1]

    cm = confusion_matrix(y_true=true_values, y_pred=predicted_values)

    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    plt.show()

    cm_true = cm / cm.astype(float).sum(axis=1)
    cm_true_percentage = cm_true * 100
    plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
                                     title="Confusion Matrix (%) - Seq2Label")
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

    pdf.create_table(table_data=data, title='Metrics final test users', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output('seq2label_metrics.pdf')

