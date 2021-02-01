from typing import Callable

import numpy as np
import pandas as pd
from sklearn.svm import SVC


class SVM:
    def __init__(self, **kwargs):
        self.svm = SVC(**kwargs)

    def fit(self, x_true, y_true):
        return self.svm.fit(x_true, y_true)

    def predict(self, x_pred):
        y_pred = self.svm.predict_proba(x_pred)
        return y_pred


def create_svm(train_epochs: int) -> SVM:
    return SVM(kernel="linear", probability=True, max_iter=train_epochs)


def train(
    train_path: str,
    val_path: str,
    model_h5_path: str,
    model_factory: Callable,
    cast_dataset: Callable,
    force: bool = False,
    train_epochs: int = 1000,
    train_batch_size: int = 128,
    binary: bool = True,
    root_dir: str = "",
) -> SVM:
    print("training svm model")
    train_df = pd.read_csv(f"{root_dir}/{train_path}")
    x_train, y_train = cast_dataset(train_df, binary=binary)

    model = model_factory(train_epochs=train_epochs)
    model.fit(x_train, np.argmax(y_train, axis=-1))

    return model, None
