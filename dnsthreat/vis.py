from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


SAVE_KW = dict(bbox_inches="tight", dpi=600)


def render_roc(
    y_true: np.array, y_pred: np.array, label: str, klass: int, pos_label: int = 1
):
    """Render ROC-curve based on the predicted resluts of the model.

    Apart of the ROC-curve plot contains an ephemeral model of
    equiprobable guessing of the resulting class.
    """
    _, ax = plt.subplots()

    # Calculate ROC for each class.
    y_true, y_pred = y_true[:, klass], y_pred[:, klass]

    fpr, tpr, thrs = roc_curve(y_true, y_pred, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    label = "%s (AUC = %05.4f)" % (label, roc_auc)

    ax.plot(
        fpr,
        tpr,
        c="black",
        label=label,
        linestyle="-",
        marker=".",
        markersize=4,
    )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray", label="Гадание")

    ax.legend(loc="lower right")

    ax.set_xlabel("Ложноположительный")
    ax.set_ylabel("Истинно положительный")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


def render_history(history: Dict[str, np.array]):
    """Render plot with training history metrics.

    The first returned plot contains loss and validation loss by iteration.
    The second returned plot contains training and validation accuracy.
    """
    _, ax = plt.subplots()
    ax.plot(
        history["loss"][1:],
        linestyle="--",
        label="Ошибка обучения",
        c="black",
    )
    ax.plot(history["val_loss"][1:], linestyle="-", label="Ошибка валидации", c="black")

    ax.set_xlabel("Итерация обучения")
    ax.set_ylabel("Ошибка")
    # ax.set_ylim(ymin=0.7)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(loc="upper right")

    _, acc = plt.subplots()
    acc.plot(
        history["accuracy"][1:],
        linestyle="--",
        label="Точность обучения",
        c="black",
    )
    acc.plot(
        history["val_accuracy"][1:],
        linestyle="-",
        label="Точность валидации",
        c="black",
    )

    acc.set_xlabel("Итерация обучения")
    acc.set_ylabel("Точность")
    acc.set_ylim(ymin=0.7)

    acc.spines["right"].set_visible(False)
    acc.spines["top"].set_visible(False)
    acc.legend(loc="lower right")

    return ax, acc


def heatmap(data, row_labels, col_labels, ax=None, **kwargs):
    """Create a heatmap from a numpy array and two lists of labels."""

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, cmap="binary", **kwargs)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="left", rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """Add annotations to the heatmap."""

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
