import pathlib
import pickle
from typing import Callable, Dict, Optional

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM,
    AlphaDropout,
    Bidirectional,
    Concatenate,
    Conv1D,
    Convolution1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    Reshape,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from wandb.keras import WandbCallback

from deeplookup.datasets import en2vec


def name_from_path(path: str) -> str:
    """Generate a model name from the H5 path."""
    name = pathlib.Path(path)
    return name.name[: -len(name.suffix)]


def history_from_path(
    model_h5_path: str, root_dir: str = "", missing_ok: bool = True
) -> Optional[Dict]:
    """Loads history from the given model path.

    Raises ValueError, when log is not presented. Error is supressed, when `mising_ok`
    is set to True.
    """
    root_dir = f"{root_dir}/" if root_dir else ""
    model_log_path = pathlib.Path(f"{root_dir}{model_h5_path}.log")

    if model_log_path.exists():
        with model_log_path.open(mode="rb") as f:
            return pickle.load(f)
    if not missing_ok:
        raise ValueError(f"No history found at '{model_log_path}'")


def create_cnn(num_classes: int = 2) -> tf.keras.Model:
    x = Input(shape=(256,), dtype="int64")
    h = Embedding(en2vec.corpus_size + 1, 128, input_length=256)(x)

    conv1 = Convolution1D(filters=256, kernel_size=10, activation="tanh")(h)
    conv2 = Convolution1D(filters=256, kernel_size=7, activation="tanh")(h)
    conv3 = Convolution1D(filters=256, kernel_size=5, activation="tanh")(h)
    conv4 = Convolution1D(filters=256, kernel_size=3, activation="tanh")(h)

    h = Concatenate()(
        [
            GlobalMaxPooling1D()(conv1),
            GlobalMaxPooling1D()(conv2),
            GlobalMaxPooling1D()(conv3),
            GlobalMaxPooling1D()(conv4),
        ]
    )

    h = Dense(1024, activation="selu", kernel_initializer="lecun_normal")(h)
    h = AlphaDropout(0.1)(h)
    h = Dense(1024, activation="selu", kernel_initializer="lecun_normal")(h)
    h = AlphaDropout(0.1)(h)

    y = Dense(num_classes, activation="softmax")(h)

    model = Model(x, y)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", AUC()]
    )
    return model


def create_rnn(num_classes: int = 2) -> tf.keras.Model:
    model = Sequential(name="dns_rnn")
    model.add(Embedding(en2vec.corpus_size + 1, 128, input_length=256))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(1024, activation="selu", kernel_initializer="lecun_normal"))
    model.add(AlphaDropout(0.1))
    model.add(Dense(1024, activation="selu", kernel_initializer="lecun_normal"))
    model.add(AlphaDropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", AUC()]
    )
    return model


def create_woodbridge_rnn(num_classes: int = 2) -> tf.keras.Model:
    model = Sequential(name="dga_woodbridge")
    model.add(Embedding(en2vec.corpus_size + 1, 128, input_length=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", AUC()]
    )
    return model


def create_vosoughi_rcnn(num_classes: int = 2) -> tf.keras.Model:
    model = Sequential(name="dns_rcnn")
    model.add(Embedding(en2vec.corpus_size + 1, 128, input_length=256))
    model.add(Conv1D(filters=128, kernel_size=8, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Reshape((1, 128)))
    model.add(LSTM(128))
    model.add(Dense(1024, activation="selu", kernel_initializer="lecun_normal"))
    model.add(AlphaDropout(0.1))
    model.add(Dense(1024, activation="selu", kernel_initializer="lecun_normal"))
    model.add(AlphaDropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", AUC()]
    )
    return model


def create_rcnn(
    num_classes: int = 2,
    corpus_size: int = en2vec.corpus_size,
    input_name: str = "input_1",
    bidir: bool = False,
) -> tf.keras.Model:
    x = Input(shape=(256,), dtype="int64", name=input_name)
    em1 = Embedding(corpus_size + 1, 128, input_length=256)(x)

    conv1 = Convolution1D(filters=64, kernel_size=2)(em1)
    conv2 = Convolution1D(filters=64, kernel_size=3)(em1)
    conv3 = Convolution1D(filters=64, kernel_size=4)(em1)
    conv4 = Convolution1D(filters=64, kernel_size=5)(em1)
    conv5 = Convolution1D(filters=64, kernel_size=6)(em1)

    pool1 = GlobalMaxPooling1D()(conv1)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool3 = GlobalMaxPooling1D()(conv3)
    pool4 = GlobalMaxPooling1D()(conv4)
    pool5 = GlobalMaxPooling1D()(conv5)
    conv = Concatenate(axis=-1)([pool1, pool2, pool3, pool4, pool5])

    conv = Dropout(0.5)(conv)
    conv = Dense(64, activation="relu")(conv)
    conv = Dropout(0.5)(conv)

    lstm = Bidirectional(LSTM(128))(em1) if bidir else LSTM(128)(em1)

    lstm = Dropout(0.5)(lstm)
    lstm = Dense(128)(lstm)

    input = Concatenate(axis=-1)([conv, lstm])
    input = Dropout(0.5)(input)

    h = Dense(128, activation="relu", kernel_initializer="lecun_normal")(input)
    h = Dropout(0.5)(h)
    y = Dense(num_classes, activation="softmax")(h)

    model = Model(x, y)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", AUC()]
    )
    return model


def create_ffnn(num_classes: int = 2):
    model = Sequential(name="dns_ffw")
    model.add(Dense(128, activation="relu", input_dim=18))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="nadam",
        metrics=["accuracy", AUC()],
    )
    return model


def train(
    train_path: str,
    val_path: str,
    model_h5_path: str,
    model_factory: Callable,
    cast_dataset: Callable,
    force: bool = False,
    train_epochs: int = 10,
    train_batch_size: int = 128,
    binary: bool = True,
    root_dir: str = "",
) -> tf.keras.Model:
    """Train the Keras model using the given datasets.

    Return either a new trained model or previously trained model.
    When `force` is set, the model will be retrained automatically.
    """
    model_h5_path = pathlib.Path(f"{root_dir}/{model_h5_path}")
    model_log_path = pathlib.Path(f"{model_h5_path}.log")

    if not force and model_h5_path.exists():
        print(f"loading existing model from {model_h5_path}")

        history = history_from_path(model_h5_path, missing_ok=False)
        return tf.keras.models.load_model(str(model_h5_path)), history

    train_df = pd.read_csv(f"{root_dir}/{train_path}")
    x_train, y_train = cast_dataset(train_df, binary=binary)

    val_df = pd.read_csv(f"{root_dir}/{val_path}")
    x_val, y_val = cast_dataset(val_df, binary=binary)

    print("training a new model")
    model = model_factory()
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping("val_loss", patience=5)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=train_epochs,
        batch_size=train_batch_size,
        callbacks=[early_stopping, WandbCallback()],
    )

    # Save the trained model and the traing history to plot the results.
    model.save(model_h5_path)
    with model_log_path.open(mode="wb") as f:
        pickle.dump(history.history, f)

    return model, history.history
