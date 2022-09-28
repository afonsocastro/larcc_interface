#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
from config.definitions import ROOT_DIR
from itertools import product
import numpy as np
from larcc_classes.documentation.PDF import PDF


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

    pull = {"true_positive": [], "false_positive": [], "false_negative": []}
    push = {"true_positive": [], "false_positive": [], "false_negative": []}
    shake = {"true_positive": [], "false_positive": [], "false_negative": []}
    twist = {"true_positive": [], "false_positive": [], "false_negative": []}

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



    data = [
        ["First name", "Last name", "Age", "City", ],  # 'testing','size'],
        ["Jules", "Smith", "34", "San Juan", ],  # 'testing','size'],
        ["Mary", "Ramos", "45", "Orlando", ],  # 'testing','size'],
        ["Carlson", "Banks", "19", "Los Angeles", ],  # 'testing','size'],
        ["Lucas", "Cimon", "31", "Saint-Mahturin-sur-Loire", ],  # 'testing','size'],
    ]

    data_as_dict = {"First name": ["Jules", "Mary", "Carlson", "Lucas"],
                    "Last name": ["Smith", "Ramos", "Banks", "Cimon"],
                    "Age": [34, '45', '19', '31']}

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Times", size=10)


    pdf.create_table(table_data=data, title='I\'m the first title', cell_width='even')
    pdf.ln()

    pdf.create_table(table_data=data, title='I start at 25', cell_width='uneven', x_start=25)
    pdf.ln()

    pdf.create_table(table_data=data, title="I'm in the middle", cell_width=22, x_start='C')
    pdf.ln()

    pdf.create_table(table_data=data_as_dict, title='Is my text red', align_header='R', align_data='R',
                     cell_width=[15, 15, 10, 45, ], x_start='C', emphasize_data=['45', 'Jules'], emphasize_style='BIU',
                     emphasize_color=(255, 0, 0))
    pdf.ln()

    pdf.output('table_class.pdf')

    exit(0)
    # with open("just.json", "w") as write_file:
    #     json.dump(loss, write_file)

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
    plt.plot(contribution_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'attendance'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(mean_loss)
    plt.plot(mean_val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # plt.show()
    plt.savefig(ROOT_DIR + "/neural_networks/keras/training_testing_n_times/training_curves_mean.png",
                bbox_inches='tight')

    # -------------------------------------------------------------------------------------------------------------
    # CONFUSION MATRIX-----------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    blues = plt.cm.Blues
    cm_mean = np.empty((n_labels, n_labels))
    for i in range(0, cm_cumulative_percentage.shape[0]):
        for j in range(0, cm_cumulative_percentage.shape[1]):
            cm_mean[i][j] = cm_cumulative_percentage[i][j] / n_times

    cm_mean = cm_mean*100
    # print("cm_mean")
    # print(cm_mean)
    #
    # print("cm_cumulative")
    # print(cm_cumulative)

    plot_confusion_matrix_percentage(confusion_matrix=cm_mean, display_labels=labels, cmap=blues,
                                     title="Mean Percentage CM (%)", decimals=.2)
    plt.savefig(ROOT_DIR + "/neural_networks/keras/training_testing_n_times/confusion_matrix_mean_percentage.png",
                bbox_inches='tight')

    plot_confusion_matrix_percentage(confusion_matrix=cm_cumulative, display_labels=labels, cmap=blues,
                                     title="Cumulative CM (%d times)" % n_times, decimals=.0)
    plt.savefig(ROOT_DIR + "/neural_networks/keras/training_testing_n_times/confusion_matrix_cumulative.png",
                bbox_inches='tight')
    # -------------------------------------------------------------------------------------------------------------
