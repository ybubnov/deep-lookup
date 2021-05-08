import pytest

from deeplookup import datasets, metrics
from deeplookup.experiments import hmm


@pytest.mark.parametrize(
    "predict, to_categorical",
    [
        (hmm.rsamp.predict_all, False),
        (hmm.bossvs.predict, True),
        (hmm.saxvsm.predict, True),
    ],
    ids=["rsamp", "bossvs", "saxvsm"],
)
def test_benchmark(predict, to_categorical, benchmark):
    x_test, _ = datasets.load_ts(200, to_categorical=to_categorical)

    benchmark.pedantic(
        predict,
        args=(x_test,),
        iterations=10,
        rounds=100,
    )
