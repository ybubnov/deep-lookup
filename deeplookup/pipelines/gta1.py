from pathlib import Path
from typing import Text

import absl
from tfx.components import (
    ImportExampleGen,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2

from deeplookup.datasets.gta1 import Gta1


_pipeline_name = "gta1"

_tfx_root = Path("tfx")
_pipeline_path = _tfx_root / "pipelines" / _pipeline_name


_data_path = _pipeline_path / "data"

# Python module files to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = Path(__file__).parent / "gta1_trainer.py"

# Path which can be listened to by the model server. Pusher will output the
# trained model here.
_serving_model_path = _pipeline_path / "serving_model" / _pipeline_name

# Sqlite ML-metadata db path.
_metadata_path = _tfx_root / "metadata" / _pipeline_name / "metadata.db"


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Path,
    module_file: Path,
    serving_model_path: Path,
    metadata_path: Path,
    data_path: Path,
) -> pipeline.Pipeline:
    builder = Gta1()
    builder.download_and_prepare()

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="*.tfrecord-[0-9]*"),
        ],
    )

    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ],
        ),
    )

    # Bring the data in to the pipeline.
    example_gen = ImportExampleGen(
        input_base=builder.data_dir,
        input_config=input_config,
        output_config=output_config,
    )

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        module_file=str(module_file),
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        materialize=True,
    )

    # Uses user-provided Python function that trains a model.
    trainer = Trainer(
        module_file=str(module_file),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500),
    )

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs["model"],
        # model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=str(serving_model_path),
            ),
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=str(pipeline_root),
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            trainer,
            pusher,
        ],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            str(metadata_path),
        ),
        enable_cache=True,
    )


def run_pipeline() -> None:
    pipeline = create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_path,
        module_file=_module_file,
        serving_model_path=_serving_model_path,
        metadata_path=_metadata_path,
        data_path=_data_path,
    )

    LocalDagRunner().run(pipeline)


if __name__ == "__main__":
    absl.logging.set_verbosity(absl.logging.INFO)
    run_pipeline()
