train_umudga = partial(
    train_keras_model,
    train_path="umudga/umudga-b-1000-train.csv",
    val_path="umudga/umudga-b-1000-val.csv",
    cast_dataset=cast_umudga,
)

train_umudga_m = partial(
    train_keras_model,
    train_path="umudga/umudga-m-1000-train.csv",
    val_path="umudga/umudga-m-1000-val.csv",
    cast_dataset=cast_umudga,
    binary=False,
)
