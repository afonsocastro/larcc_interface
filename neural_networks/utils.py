#!/usr/bin/env python3

from itertools import product
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix_percentage(confusion_matrix, display_labels=None, cmap="viridis",
                                     xticks_rotation="horizontal", title="Confusion Matrix"):
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
        text_cm = format(cm[i, j], ".1f")
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
