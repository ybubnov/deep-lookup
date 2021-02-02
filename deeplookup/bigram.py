from typing import Callable, List

import nltk as nl
import numpy as np
import pandas as pd
from scipy.special import softmax


class Bigram:
    def __init__(self):
        self.freqs = {}

    @staticmethod
    def str_to_bigrams(st: str) -> List[str]:
        bigrams = nl.bigrams(st)
        bigrams = map(lambda x: "".join(x), bigrams)
        return list(bigrams)

    def score_normal(self, qname: str) -> float:
        bigrams = self.str_to_bigrams(qname)
        return sum([self.freqs.get(bg, 0) ** 2 for bg in bigrams])

    def score_random(self, qname: str) -> float:
        bigrams = self.str_to_bigrams(qname)
        return sum([self.freqs.get(bg, 0) for bg in bigrams]) / len(bigrams)

    def fit(self, x_true: List[str], y=None) -> None:
        for x in x_true:
            bigrams = self.str_to_bigrams(x)
            for bg in bigrams:
                self.freqs[bg] = self.freqs.get(bg, 0) + 1

        for bg in self.freqs:
            self.freqs[bg] /= len(self.freqs)

    def predict(self, x_pred) -> np.array:
        y_pred = []
        for x in x_pred:
            y_pred.append(softmax([self.score_random(x), self.score_normal(x)]))
        return np.array(y_pred)


def create_bigram(train_epochs: int):
    return Bigram()


def train(
    train_path: str,
    val_path: str,
    model_h5_path: str,
    cast_dataset: Callable,
    model_factory: Callable = create_bigram,
    force: bool = False,
    train_epochs: int = 1000,
    train_batch_size: int = 128,
    binary: bool = True,
    root_dir: str = "",
):
    print("training bigram model")
    train_df = pd.read_csv(f"{root_dir}/{train_path}")
    x_train, y_train = cast_dataset(train_df, binary=binary, encode=False)

    model = model_factory(train_epochs=train_epochs)
    model.fit(x_train, np.argmax(y_train, axis=-1))

    return model, None
