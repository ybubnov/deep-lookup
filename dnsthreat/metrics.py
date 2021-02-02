from typing import Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def eval_class(y_pred: np.array, y_true: np.array, average: str = "binary") -> None:
    """Print metrics and confusion matrix for the given predictions."""
    print(
        f"accuracy: {accuracy_score(y_true, y_pred):9.4f} - "
        f"precision: {precision_score(y_true, y_pred, average=average):9.4f} - "
        f"recall: {recall_score(y_true, y_pred, average=average):9.4f} - ",
        f"f1: {f1_score(y_true, y_pred, average=average):9.4f}",
        end=" - ",
    )

    if average == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(
            f"true_positives: {tp} - true_negatives: {tn} - "
            f"false_positives: {fp} - false_negatives: {fn}"
        )
    else:
        print()


def eval(
    model: tf.keras.Model,
    test_path: str,
    cast_dataset: Callable,
    binary: bool = True,
    prepare_predict: bool = True,
    prepare_skip: bool = False,
    average: str = "binary",
    root_dir: str = "",
    encode: bool = True,
) -> Tuple[np.array, np.array]:
    """Evaluate the model on the test dataset.

    Method prints the basic metrics of the model and returns true and predicted
    results of the test data classification.
    """
    test_df = pd.read_csv(f"{root_dir}/{test_path}")
    x_test, y_test = cast_dataset(test_df, binary=binary, encode=encode)

    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=-1) if prepare_predict else y_prob

    if prepare_skip:
        return y_test, y_prob

    # Convert from categorical to numerical represetnation of the label.
    if binary:
        y_true = np.argmax(y_test, axis=-1)
        eval_class(y_pred, y_true, average=average)
    else:
        y_pred = to_categorical(y_pred)
        for cls in range(y_pred.shape[1]):
            print(f"metrics for class {cls}:")
            y_pr = y_pred[:, cls].astype("int64")
            y_true = y_test[:, cls].astype("int64")
            eval_class(y_pr, y_true)

    return y_test, y_prob
