from datetime import date
from functools import partial
from typing import Callable

import numpy as np
import wandb

from deeplookup import bigram, datasets, metrics, nn, svm


ROOT_DIR = "csv-datasets"

B_MODELS = [
    (
        partial(svm.train, **datasets.dtqbc_b.train),
        partial(metrics.eval, **datasets.dtqbc_b.test),
        svm.create_svm,
        "model/b-tun-svm.h5",
        {},
        [],
    ),
    (
        partial(bigram.train, **datasets.dtqbc2_b.train),
        partial(metrics.eval, **datasets.dtqbc2_b.test, encode=False),
        bigram.create_bigram,
        "model/b-tun-bigram.h5",
        {},
        [],
    ),
    (
        partial(nn.train, **datasets.dtqbc2_b.train),
        partial(metrics.eval, **datasets.dtqbc2_b.test),
        nn.create_cnn,
        "model/b-tun-cnn.h5",
        {"train_epochs": 20},
        [
            ("global_max_pooling1d", "Слой пулинга (10)"),
            ("global_max_pooling1d_1", "Слой пулинга (7)"),
            ("global_max_pooling1d_2", "Слой пулинга (5)"),
            ("global_max_pooling1d_3", "Слой пулинга (3)"),
        ],
    ),
    (
        partial(nn.train, **datasets.dtqbc2_b.train),
        partial(metrics.eval, **datasets.dtqbc2_b.test),
        nn.create_rnn,
        "model/b-tun-rnn.h5",
        {"train_epochs": 20},
        [("lstm_1", "LSTM слой")],
    ),
    (
        partial(nn.train, **datasets.dtqbc_b.train),
        partial(metrics.eval, **datasets.dtqbc_b.test),
        nn.create_ffnn,
        "model/b-tun-ffnn.h5",
        {"train_epochs": 300, "force": True},
        [],
    ),
]

M_MODELS = [
    (
        partial(svm.train, **datasets.dtqbc_m.train),
        partial(metrics.eval, **datasets.dtqbc_m.test),
        svm.create_svm,
        "model/m-tun-svm.h5",
        {"train_epochs": 10000, "force": True},
    ),
    (
        partial(nn.train, **datasets.dtqbc2_m.train),
        partial(metrics.eval, **datasets.dtqbc2_m.test),
        partial(nn.create_cnn, num_classes=5),
        "model/m-tun-cnn.h5",
        {"train_epochs": 20, "force": True},
    ),
    (
        partial(nn.train, **datasets.dtqbc2_m.train),
        partial(metrics.eval, **datasets.dtqbc2_m.test),
        partial(nn.create_rnn, num_classes=5),
        "model/m-tun-rnn.h5",
        {"train_epochs": 20, "force": True},
    ),
    (
        partial(nn.train, **datasets.dtqbc_m.train),
        partial(metrics.eval, **datasets.dtqbc_m.test),
        partial(nn.create_ffnn, num_classes=5),
        "model/m-tun-ffnn.h5",
        {"train_epochs": 300, "force": True},
    ),
]


CLASS_NAMES = ["normal", "dns2tcp", "dnscapy", "iodine", "tuns"]


def fit(train, evaluate, factory, h5_path, kw, layers) -> Callable:
    kw["root_dir"] = ROOT_DIR
    model, _ = train(model_factory=factory, model_h5_path=h5_path, **kw)
    return partial(evaluate, model, root_dir=ROOT_DIR)


def fit_multilabel(train, evaluate, factory, h5_path, kw) -> Callable:
    kw["root_dir"] = ROOT_DIR
    model, _ = train(model_factory=factory, model_h5_path=h5_path, **kw)
    return partial(evaluate, model, root_dir=ROOT_DIR)


def main():
    today = date.today()

    for train, evaluate, factory, h5_path, kw, layers in B_MODELS:
        model_name = nn.name_from_path(h5_path)
        model_run = wandb.init(
            project="deep-lookup",
            group="tunneling-detection",
            name=f"{today}-{model_name}",
            tags=["tunneling", "binary"],
            config=dict(model=model_name, **kw),
            reinit=True,
        )

        with model_run:
            fit(train, evaluate, factory, h5_path, kw, layers)()

    for train, evaluate, factory, h5_path, kw in M_MODELS:
        model_name = nn.name_from_path(h5_path)

        model_run = wandb.init(
            project="deep-lookup",
            group="tunneling-categorization",
            name=f"{today}-{model_name}",
            tags=["tunneling", "multilabel"],
            config=dict(model=model_name, **kw),
            reinit=True,
        )

        with model_run:
            y_true, y_pred = fit_multilabel(train, evaluate, factory, h5_path, kw)()
            y_true, y_pred = np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)

            conf_mat = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=CLASS_NAMES,
            )

            wandb.log({"conf_mat": conf_mat})


if __name__ == "__main__":
    main()
