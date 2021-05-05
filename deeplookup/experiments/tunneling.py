import pathlib
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from deeplookup import bigram, datasets, metrics, nn, svm, vis
from deeplookup.metrics import confusion_matrix


ROOT_DIR = "csv-datasets"

B_MODELS = [
    (
        partial(svm.train, **datasets.irdtun_b.train),
        partial(metrics.eval, **datasets.irdtun_b.test),
        svm.create_svm,
        "model/b-tun-svm.h5",
        {},
        [],
    ),
    (
        partial(bigram.train, **datasets.irdtun2_b.train),
        partial(metrics.eval, **datasets.irdtun2_b.test, encode=False),
        bigram.create_bigram,
        "model/b-tun-bigram.h5",
        {},
        [],
    ),
    (
        partial(nn.train, **datasets.irdtun2_b.train),
        partial(metrics.eval, **datasets.irdtun2_b.test),
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
        partial(nn.train, **datasets.irdtun2_b.train),
        partial(metrics.eval, **datasets.irdtun2_b.test),
        nn.create_rnn,
        "model/b-tun-rnn.h5",
        {"train_epochs": 20},
        [("lstm_1", "LSTM слой")],
    ),
    (
        partial(nn.train, **datasets.irdtun_b.train),
        partial(metrics.eval, **datasets.irdtun_b.test),
        nn.create_ffnn,
        "model/b-tun-ffnn.h5",
        {"train_epochs": 300},
        [],
    ),
]

M_MODELS = [
    (
        partial(svm.train, **datasets.irdtun_m.train),
        partial(metrics.eval, **datasets.irdtun_m.test),
        svm.create_svm,
        "model/m-tun-svm.h5",
        {"train_epochs": 10000},
    ),
    (
        partial(nn.train, **datasets.irdtun2_m.train),
        partial(metrics.eval, **datasets.irdtun2_m.test),
        partial(nn.create_cnn, num_classes=5),
        "model/m-tun-cnn.h5",
        {"train_epochs": 20},
    ),
    (
        partial(nn.train, **datasets.irdtun2_m.train),
        partial(metrics.eval, **datasets.irdtun2_m.test),
        partial(nn.create_rnn, num_classes=5),
        "model/m-tun-rnn.h5",
        {"train_epochs": 20},
    ),
    (
        partial(nn.train, **datasets.irdtun_m.train),
        partial(metrics.eval, **datasets.irdtun_m.test),
        partial(nn.create_ffnn, num_classes=5),
        "model/m-tun-ffnn.h5",
        {"train_epochs": 300},
    ),
]


CLASS_NAMES = ["normal", "dns2tcp", "dnscapy", "iodine", "tuns"]


def fit(train, evaluate, factory, h5_path, kw, layers):
    model_name = pathlib.Path(h5_path)
    model_name = model_name.name[: -len(model_name.suffix)]

    kw["root_dir"] = ROOT_DIR
    model, history = train(model_factory=factory, model_h5_path=h5_path, **kw)
    return partial(evaluate, model, root_dir=ROOT_DIR)


def fit_multilabel(train, evaluate, factory, h5_path, kw):
    model_name = pathlib.Path(h5_path)
    model_name = model_name.name[: -len(model_name.suffix)]

    kw["root_dir"] = ROOT_DIR
    model, history = train(model_factory=factory, model_h5_path=h5_path, **kw)
    return partial(evaluate, model, root_dir=ROOT_DIR)


def main():
    for train, evaluate, factory, h5_path, kw, layers in B_MODELS:
        y_true, y_pred = fit(train, evaluate, factory, h5_path, kw, layers)()

        ax0 = vis.render_roc(y_true, y_pred, klass=0, label="Безопасный DNS")
        ax0.figure.savefig(f"{ROOT_DIR}/images/{model_name}-roc-0.png", **vis.SAVE_KW)

        ax1 = vis.render_roc(y_true, y_pred, klass=1, label="Небезопасный DNS")
        ax1.figure.savefig(f"{ROOT_DIR}/images/{model_name}-roc-1.png", **vis.SAVE_KW)

        if history:
            ax_hist, acc_hist = vis.render_history(history)
            ax_hist.figure.savefig(
                f"{ROOT_DIR}/images/{model_name}-hist.png", **vis.SAVE_KW
            )
            acc_hist.figure.savefig(
                f"{ROOT_DIR}/images/{model_name}-acc.png", **vis.SAVE_KW
            )

    for train, evaluate, factory, h5_path, kw in M_MODELS:
        y_true, y_pred = fit_multilabel(train, evaluate, factory, h5_path, kw)()
        print("-" * 80)

        cm = confusion_matrix(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1))

        heatmap_fig, ax = plt.subplots()
        hm = vis.heatmap(cm, CLASS_NAMES, CLASS_NAMES, ax=ax)
        vis.annotate_heatmap(hm, valfmt="{x}")
        heatmap_fig.savefig(
            f"{ROOT_DIR}/images/{model_name}-heatmap.png", **vis.SAVE_KW
        )


if __name__ == "__main__":
    main()
