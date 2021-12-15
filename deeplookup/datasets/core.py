import random
import string
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tensorflow.keras.utils import to_categorical


class str2vec:
    """Encode string to the vector of integers using an alphabet."""

    EN_US_ALPHABET = string.printable

    TL_DOMAINS = ["tunnel.tuns.org.", "hidemyself.org.", "tunnel.example.org."]

    def __init__(self, alphabet: str, maxlen: int = 256) -> None:
        self.corpus = {ch: pos + 1 for pos, ch in enumerate(alphabet)}
        self.alphabet = alphabet
        self.maxlen = maxlen

    @property
    def corpus_size(self):
        return len(self.corpus)

    def convert(self, s: bytes) -> np.array:
        for d in self.TL_DOMAINS:
            s = s.replace(d, "")

        maxlen = min(len(s), self.maxlen)
        str2idx = np.zeros(self.maxlen, dtype="int64")

        for i in range(1, maxlen + 1):
            c = s[-i]
            str2idx[i - 1] = self.corpus[c] if c in self.corpus else 0
        return str2idx

    def decode(self, i: int) -> str:
        return self.alphabet[i - 1] if 0 < i < len(self.alphabet) else " "


# Define encoder for English/US alphabet.
en2vec = str2vec(str2vec.EN_US_ALPHABET)


def cast_umudga(
    df: pd.DataFrame,
    binary: bool = False,
    pos_class: str = "normal",
    encode: bool = True,
) -> pd.DataFrame:
    """Prepare the UMUDGA dataset for training a model.

    The dataset contains domain names generated using various algorithms that
    are exploit within bot-nets and malicious software.
    """
    df = df.sample(frac=1.0)

    df.loc[df["class"] == "legit", "class"] = pos_class

    if binary:
        # When the task data must be prepared for a binary problem,
        # override the "class" of the input data with negated binary class.
        #
        # After this operation, all classes will divided into 2 categories.
        df.loc[df["class"] != pos_class, "class"] = f"~{pos_class}"

    # Encode the string representation of the domain name into
    # the integer vector, which will be used as inputs of the model.
    df["x_true"] = df["domain"].map(en2vec.convert)

    # Assign to each class a code of the category, these values
    # will be used as outputs of the model.
    categories = sorted(pd.unique(df["class"].to_numpy().ravel()))
    class_dtype = CategoricalDtype(categories=categories, ordered=True)

    df["y_true"] = df["class"].astype(class_dtype).cat.codes

    x_true = df["x_true"].to_numpy().tolist()
    y_true = to_categorical(df["y_true"].to_numpy()).tolist()
    return np.asarray(x_true, dtype="int64"), np.asarray(y_true)


def cast_dtqbc(df, binary: bool = False, pos_class: int = 0, encode: bool = True):
    """Prepare IRDTUN dataset for training a model.

    The dataset contains domain names and other DNS-packet attributes from
    various tunneling programs, like `iodine`, `dnstun`, `tuns`, `dns2tcp`.
    """
    df = df.sample(frac=1.0)
    x_true = df.drop(columns=["qname", "label"]).to_numpy()

    if binary:
        df.loc[df["label"] != pos_class, "label"] = 1 - pos_class

    y_true = to_categorical(df["label"].to_numpy())
    return x_true, y_true


def cast_dtqbc2(
    df: pd.DataFrame, binary: bool = False, pos_class: int = 0, encode=True
) -> pd.DataFrame:
    """Prepare IRDTUN-2 dataset for training a model.

    The dataset contains domain names produced by various tunneling programs,
    like `iodine`, `dnstun`, `tuns`, `dns2tcp`.

    The dataset is used for domain names text analysis.
    """
    df = df.sample(frac=1.0)
    if binary:
        df.loc[df["label"] != pos_class, "label"] = 1 - pos_class

    if encode:
        df["x_true"] = df["qname"].map(en2vec.convert)
    else:
        df["x_true"] = df["qname"]
    df["y_true"] = df["label"]

    x_true = df["x_true"].to_numpy().tolist()
    y_true = to_categorical(df["y_true"].to_numpy()).tolist()
    x_true = np.asarray(x_true, dtype="int64") if encode else np.asarray(x_true)

    return x_true, np.asarray(y_true)


def load_ts(
    num: int = 100,
    window_ratio: float = 0.1,
    window_prob: float = 0.5,
    to_categorical: bool = False,
    noise_ratio: int = 0.2,
):
    """Create a dataset for timeseries classification problem.

    Change window ratio and probability to get the different behavior
    modelling for DNS threats.
    """
    window_len = int(num * window_ratio)
    probs = [random.random() % window_prob for _ in range(num)]

    x_train, y_train = [], []
    assembly_num = 1 if window_len == 0 else num - window_len + 1

    for i in range(assembly_num):
        x = probs.copy()
        y = [0] * len(x)

        for j in range(window_len):
            x[i + j] = window_prob + (random.random() % (1 - window_prob))
            y[i + j] = 1

        x_train.append(x)
        if to_categorical:
            y = 1 if window_ratio > noise_ratio else 0

        y_train.append(y)

    return np.array(x_train), np.array(y_train)


def load_train_ts(num: int = 200) -> Tuple[np.array, np.array]:
    x_train, y_train = [], []

    for w in np.arange(0.0, 0.4, 0.05):
        x, y = load_ts(num, window_ratio=w, to_categorical=True)
        x_train.append(x)
        y_train.append(y)

    return np.concatenate(x_train), np.concatenate(y_train)


@dataclass
class Dataset:
    train_path: str
    val_path: str
    test_path: str
    cast_dataset: Callable
    binary: bool = field(default=True)

    @property
    def train(self):
        return {
            "train_path": self.train_path,
            "val_path": self.val_path,
            "cast_dataset": self.cast_dataset,
            "binary": self.binary,
        }

    @property
    def test(self):
        return {
            "test_path": self.test_path,
            "cast_dataset": self.cast_dataset,
            "binary": self.binary,
        }


dtqbc_b = Dataset(
    train_path="dtqbc/dtqbc-b-train.csv",
    val_path="dtqbc/dtqbc-b-val.csv",
    test_path="dtqbc/dtqbc-b-test.csv",
    cast_dataset=cast_dtqbc,
)

dtqbc2_b = Dataset(
    train_path="dtqbc/dtqbc-b-train.csv",
    val_path="dtqbc/dtqbc-b-val.csv",
    test_path="dtqbc/dtqbc-b-test.csv",
    cast_dataset=cast_dtqbc2,
)

dtqbc_m = Dataset(
    train_path="dtqbc/dtqbc-m-train.csv",
    val_path="dtqbc/dtqbc-m-val.csv",
    test_path="dtqbc/dtqbc-m-test.csv",
    cast_dataset=cast_dtqbc,
    binary=False,
)

dtqbc2_m = Dataset(
    train_path="dtqbc/dtqbc-m-train.csv",
    val_path="dtqbc/dtqbc-m-val.csv",
    test_path="dtqbc/dtqbc-m-test.csv",
    cast_dataset=cast_dtqbc2,
    binary=False,
)

umudga_b = Dataset(
    train_path="umudga/umudga-b-1000-train.csv",
    val_path="umudga/umudga-b-1000-val.csv",
    test_path="umudga/umudga-b-1000-test.csv",
    cast_dataset=cast_umudga,
)

umudga_m = Dataset(
    train_path="umudga/umudga-m-1000-train.csv",
    val_path="umudga/umudga-m-1000-val.csv",
    test_path="umudga/umudga-m-1000-test.csv",
    cast_dataset=cast_umudga,
    binary=False,
)
