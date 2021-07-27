from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from deeplookup import datasets, metrics, nn, vis


ROOT_DIR = "csv-datasets"

B_MODELS = [
    (nn.create_cnn, "model/b-dga-zhang.h5"),
    (nn.create_woodbridge_rnn, "model/b-dga-woodbridge.h5"),
    (nn.create_vosoughi_rcnn, "model/b-dga-vosoughi.h5"),
    (partial(nn.create_rcnn, bidir=False), "model/b-dga-highman.h5"),
    (partial(nn.create_rcnn, bidir=True), "model/b-dga-proposed.h5"),
]

M_MODELS = [
    (nn.create_cnn, "model/m-dga-zhang.h5"),
    (nn.create_woodbridge_rnn, "model/m-dga-woodbridge.h5"),
    (nn.create_vosoughi_rcnn, "model/m-dga-vosoughi.h5"),
    (partial(nn.create_rcnn, bidir=False), "model/m-dga-highman.h5"),
    (partial(nn.create_rcnn, bidir=True), "model/m-dga-proposed.h5"),
]


class_names = [
    "alureon",
    "banjori",
    "bedep",
    "chinad",
    "class",
    "cryptolocker",
    "dyre",
    "fobber_v1",
    "fobber_v2",
    "gozi_gpl",
    "gozi_luther",
    "gozi_nasa",
    "gozi_rfc4343",
    "kraken_v1",
    "kraken_v2",
    "matsnu",
    "murofet_v1",
    "murofet_v2",
    "murofet_v3",
    "normal",
    "nymaim",
    "padcrypt",
    "proslikefan",
    "pykspa",
    "pykspa_noise",
    "qakbot",
    "ramdo",
    "ramnit",
    "ranbyus_v1",
    "ranbyus_v2",
    "sisron",
    "suppobox_1",
    "suppobox_2",
    "symmi",
    "tempedreve",
    "vawtrak_v1",
    "vawtrak_v2",
    "vawtrak_v3",
]


def fit(factory: Callable, h5_path: str) -> Callable:
    model, _ = nn.train(
        model_factory=factory,
        model_h5_path=h5_path,
        train_epochs=20,
        root_dir=ROOT_DIR,
        **datasets.umudga_b.train,
    )
    return partial(metrics.eval, model, root_dir=ROOT_DIR, **datasets.umudga_b.test)


def fit_multilabel(factory: Callable, h5_path: str) -> Callable:
    model, _ = nn.train(
        model_factory=partial(factory, num_classes=len(class_names)),
        model_h5_path=h5_path,
        train_epochs=20,
        root_dir=ROOT_DIR,
        **datasets.umudga_m.train,
    )
    return partial(metrics.eval, model, root_dir=ROOT_DIR, **datasets.umudga_m.test)


def main():
    for factory, h5_path in B_MODELS:
        y_true, y_pred = fit(factory, h5_path)()
        print("-" * 80)

        model_name = nn.name_from_path(h5_path)
        model_history = nn.history_from_path(h5_path, ROOT_DIR, missing_ok=True)

        ax0 = vis.render_roc(y_true, y_pred, klass=0, label="Безопасный DNS")
        ax0.figure.savefig(f"{ROOT_DIR}/images/{model_name}-roc-0.png", **vis.SAVE_KW)

        ax1 = vis.render_roc(y_true, y_pred, klass=1, label="Небезопасный DNS")
        ax1.figure.savefig(f"{ROOT_DIR}/images/{model_name}-roc-1.png", **vis.SAVE_KW)

        if model_history:
            ax_hist, acc_hist = vis.render_history(model_history)
            ax_hist.figure.savefig(
                f"{ROOT_DIR}/images/{model_name}-hist.png", **vis.SAVE_KW
            )
            acc_hist.figure.savefig(
                f"{ROOT_DIR}/images/{model_name}-acc.png", **vis.SAVE_KW
            )

    for factory, h5_path in M_MODELS:
        y_true, y_pred = fit_multilabel(factory, h5_path)()
        y_true, y_pred = np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)

        metrics.eval_class(y_pred, y_true, average="micro")
        metrics.eval_class(y_pred, y_true, average="macro")
        print("-" * 80)

        cm = metrics.confusion_matrix(y_true, y_pred)

        heatmap_fig, ax = plt.subplots(figsize=(20, 20))
        hm = vis.heatmap(cm, class_names, class_names, ax=ax)
        vis.annotate_heatmap(hm, valfmt="{x}")

        model_name = nn.name_from_path(h5_path)
        heatmap_fig.savefig(
            f"{ROOT_DIR}/images/{model_name}-heatmap.png", **vis.SAVE_KW
        )


if __name__ == "__main__":
    main()
