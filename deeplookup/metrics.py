from typing import Callable, Optional, Tuple

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

from deeplookup import datasets


_dataset_cache = {}


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
    limit: Optional[int] = None,
) -> Tuple[np.array, np.array]:
    """Evaluate the model on the test dataset.

    Method prints the basic metrics of the model and returns true and predicted
    results of the test data classification.
    """
    path = f"{root_dir}/{test_path}"
    if path not in _dataset_cache:
        _dataset_cache[path] = pd.read_csv(path)

    dataset_df = _dataset_cache[path]
    dataset_df = dataset_df.iloc[:limit] if limit else dataset_df

    x_test, y_test = cast_dataset(dataset_df, binary=binary, encode=encode)

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


def ts_score(
    model, window_prob: float = 0.5, to_categorical: bool = True
) -> pd.DataFrame:
    experiments = []

    for window_ratio in np.arange(0.0, 1.0, 0.1):
        x_test, y_test = datasets.load_ts(
            num=200,
            window_ratio=window_ratio,
            window_prob=window_prob,
            to_categorical=to_categorical,
        )

        if to_categorical:
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred, normalize=True)
            precision = precision_score(y_test, y_pred, zero_division=0.0)
            recall = recall_score(y_test, y_pred, zero_division=0.0)
            f1 = f1_score(y_test, y_pred, zero_division=0.0)
        else:
            accuracy, precision, recall, f1 = [], [], [], []
            for x, y in zip(x_test, y_test):
                y_pred = model.predict(x)
                accuracy.append(accuracy_score(y, y_pred, normalize=True))
                precision.append(precision_score(y, y_pred, zero_division=0.0))
                recall.append(recall_score(y, y_pred, zero_division=0.0))
                f1.append(f1_score(y, y_pred, zero_division=0.0))

            accuracy = np.average(accuracy)
            precision = np.average(precision)
            recall = np.average(recall)
            f1 = np.average(f1)

        experiments.append(
            dict(
                window_ratio=window_ratio,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return pd.DataFrame(experiments)
