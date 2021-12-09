from datetime import date
from functools import partial
from typing import Callable

import numpy as np
import wandb

from deeplookup import datasets, metrics, nn


ROOT_DIR = "csv-datasets"

B_MODELS = [
    (nn.create_cnn, "model/b-dga-zhang.h5"),
    (nn.create_woodbridge_rnn, "model/b-dga-woodbridge.h5"),
    (nn.create_vosoughi_rcnn, "model/b-dga-vosoughi.h5"),
    (partial(nn.create_rcnn, bidir=False), "model/b-dga-highman.h5"),
    (partial(nn.create_rcnn, bidir=True), "model/b-dga-ybubnov.h5"),
]

M_MODELS = [
    (nn.create_cnn, "model/m-dga-zhang.h5"),
    (nn.create_woodbridge_rnn, "model/m-dga-woodbridge.h5"),
    (nn.create_vosoughi_rcnn, "model/m-dga-vosoughi.h5"),
    (partial(nn.create_rcnn, bidir=False), "model/m-dga-highman.h5"),
    (partial(nn.create_rcnn, bidir=True), "model/m-dga-ybubnov.h5"),
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
    today = date.today()

    for factory, h5_path in B_MODELS:
        model_name = nn.name_from_path(h5_path)
        model_run = wandb.init(
            project="deep-lookup",
            group="dga-detection",
            name=f"{today}-{model_name}",
            tags=["dga", "binary"],
            config=dict(model=model_name),
            reinit=True,
        )

        with model_run:
            fit(factory, h5_path)()

    for factory, h5_path in M_MODELS:
        model_name = nn.name_from_path(h5_path)
        model_run = wandb.init(
            project="deep-lookup",
            group="dga-categorization",
            name=f"{today}-{model_name}",
            tags=["dga", "multilabel"],
            config=dict(model=model_name),
            reinit=True,
        )

        with model_run:
            y_true, y_pred = fit_multilabel(factory, h5_path)()
            y_true, y_pred = np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)

            conf_mat = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )

            wandb.log({"conf_mat": conf_mat})


if __name__ == "__main__":
    main()
