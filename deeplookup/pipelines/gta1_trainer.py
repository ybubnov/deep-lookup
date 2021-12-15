from typing import Dict, List

import absl
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.tfxio import dataset_options

from deeplookup.nn import create_rcnn as build_keras_model


def _tokenize_domain(domains: tf.Tensor, sequence_length: int = 256) -> tf.Tensor:
    tokenizer = tf_text.UnicodeCharTokenizer()

    pad_token = tf.constant(0, dtype=tf.int32)

    tokens = tokenizer.tokenize(domains)
    tokens = tokens.merge_dims(1, 2)
    tokens = tokens[:, :sequence_length].to_tensor(default_value=pad_token)

    pad = sequence_length - tf.shape(tokens)[1]
    tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values=pad_token)

    tokens = tf.reshape(tokens, [-1, sequence_length])
    return tf.cast(tokens, dtype=tf.int64)


def _transform_label(labels: tf.Tensor) -> tf.Tensor:
    return tf.one_hot(tf.math.sign(labels), 2)[:, 0]


# TFX Transform will call this function.
def preprocessing_fn(
    inputs, feature_key: str = "domain", label_key: str = "class"
) -> Dict[str, tf.Tensor]:
    return {
        feature_key + "_xf_input": _tokenize_domain(inputs[feature_key]),
        label_key + "_xf": _transform_label(inputs[label_key]),
    }


def _input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 200,
) -> tf.data.Dataset:
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key="class_xf",
        ),
        tf_transform_output.transformed_metadata.schema,
    )

    return dataset.repeat()


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs) -> None:
    """Train model based on given parameters.

    Args:
      fn_args: Holds arguments used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=256,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=128,
    )

    model = build_keras_model(
        corpus_size=10_000,
        input_name="domain_xf_input",
        bidir=True,
    )
    model.summary(print_fn=absl.logging.info)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
    )

    model.save(fn_args.serving_model_dir + "/model.h5", save_format="h5")
