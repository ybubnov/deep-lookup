import pytest

from deeplookup import datasets, metrics
from deeplookup.experiments import hmm


@pytest.mark.parametrize(
    "predict",
    [hmm.rsamp.predict_proba, hmm.bossvs.predict, hmm.saxvsm.predict],
    ids=["rsamp", "bossvs", "saxvsm"],
)
def test_benchmark(predict, benchmark):
    x_test, _ = datasets.load_ts(200, to_categorical=True)
    benchmark.pedantic(predict, args=(x_test[0],), iterations=10, rounds=100)
