#!/usr/bin/env python3

from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    test_data = np.load('/tmp/test_data.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    model = keras.models.load_model("myModel")
    predictions_list = []
    pull_matx = []
    push_matx = []
    shake_matx = []
    for i in range(0, len(test_data)):
        prediction = model.predict(x=test_data[i:i+1, :-1], verbose=2)
        predict_lst = [prediction[0][0], prediction[0][1], prediction[0][2]]
        decoded_prediction = np.argmax(prediction)

        if decoded_prediction == 0:
            pull_matx.append(predict_lst)
        elif decoded_prediction == 1:
            push_matx.append(predict_lst)
        elif decoded_prediction == 2:
            shake_matx.append(predict_lst)

        predictions_list.append(decoded_prediction)

    pull_matrix = np.array(pull_matx)
    push_matrix = np.array(push_matx)
    shake_matrix = np.array(shake_matx)

    # Calculate mean values across each column
    mean_0 = pull_matrix.mean(axis=0)
    minimum_0 = np.min(pull_matrix, axis=0)
    maximum_0 = np.max(pull_matrix, axis=0)

    mean_1 = push_matrix.mean(axis=0)
    minimum_1 = np.min(push_matrix, axis=0)
    maximum_1 = np.max(push_matrix, axis=0)

    mean_2 = shake_matrix.mean(axis=0)
    minimum_2 = np.min(shake_matrix, axis=0)
    maximum_2 = np.max(shake_matrix, axis=0)

    labels = ['PULL', 'PUSH', 'SHAKE']
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, mean_0, width, label='PULL')
    rects2 = ax.bar(x, mean_1, width, label='PUSH')
    rects3 = ax.bar(x + width, mean_2, width, label='SHAKE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Confidence')
    ax.set_title('Mean of confidence scores for each interaction')
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    exit(0)

    predicted_values = np.asarray(predictions_list)

    cm = confusion_matrix(y_true=test_data[:, -1], y_pred=predicted_values)



    # cm_plot_labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    cm_plot_labels = ['PULL', 'PUSH', 'SHAKE']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_plot_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()
