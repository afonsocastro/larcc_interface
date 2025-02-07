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

    # test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    test_data_7 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Rod_learning_data_7.npy", mmap_mode=None,
                        allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_9 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Ali_learning_data_9.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_11 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Ana_learning_data_11.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_14 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Dia_learning_data_14.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_12 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Gui_learning_data_12.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_8 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Ine_learning_data_8.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_13 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Joe_learning_data_11.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_6 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Luc_learning_data_7.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_5 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Maf_learning_data_6.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_10 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Mig_learning_data_10.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    test_data_4 = np.load(ROOT_DIR + "/data_storage/data/processed_learning_data/Ru_learning_data_5.npy",
                          mmap_mode=None,
                          allow_pickle=False, fix_imports=True, encoding='ASCII')

    model = keras.models.load_model("myModel")

    metrics_per_user = []
    tests = [test_data_4, test_data_5, test_data_6, test_data_7, test_data_8, test_data_9, test_data_10, test_data_13]
    for td in range(0, len(tests)):
        name = ""
        if td == 0:
            name = "Ruben_4"
        elif td == 1:
            name = "Mafalda_5"
        elif td == 2:
            name = "Lucas_6"
        elif td == 3:
            name = "Rodrigo_7"
        elif td == 4:
            name = "Ines_8"
        elif td == 5:
            name = "Alina_9"
        elif td == 6:
            name = "Miguel_10"
        elif td == 7:
            name = "Joel_13"

        test_data = tests[td]

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

        metrics_per_user.append({"name": name, "metrics": metrics})
        # metrics_per_user.append(
        #     {"name": name, "metrics_pull": metrics_pull, "metrics_push": metrics_push, "metrics_shake": metrics_shake,
        #      "metrics_twist": metrics_twist})

    print(metrics_per_user)
    data = [
        ["Name", "Accuracy", "Precision", "Recall", "F1", ]
    ]
    for mp in metrics_per_user:
        data.append(
            [mp["name"], str(round(mp["metrics"]["accuracy"], 4)), str(round(mp["metrics"]["precision"], 4)),
             str(round(mp["metrics"]["recall"], 4)), str(round(mp["metrics"]["f1"], 4)), ]
        )

    # for mp in metrics_per_user:
    #     data.append(
    #         [mp["name"], str(round(mp["metrics_twist"]["accuracy"], 4)), str(round(mp["metrics_twist"]["precision"], 4)),
    #          str(round(mp["metrics_twist"]["recall"], 4)), str(round(mp["metrics_twist"]["f1"], 4)), ]
    #     )

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)

    pdf.create_table(table_data=data, title='Metrics Per Test User', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.output('metrics_table_per_user.pdf')
