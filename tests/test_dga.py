import pathlib

import pytest

from deeplookup.experiments import dga


kwargs = {"limit": 1, "prepare_skip": True, "prepare_predict": True}


def model_ids(models):
    return [pathlib.Path(p).stem for _, p in models]


@pytest.mark.parametrize("factory, path", dga.B_MODELS, ids=model_ids(dga.B_MODELS))
def test_binary_benchmark(factory, path, benchmark):
    function = dga.fit(factory, path)
    benchmark.pedantic(function, kwargs=kwargs, iterations=10, rounds=100)


@pytest.mark.parametrize("factory, path", dga.M_MODELS, ids=model_ids(dga.M_MODELS))
def test_multilabel_benchmark(factory, path, benchmark):
    function = dga.fit_multilabel(factory, path)
    benchmark.pedantic(function, kwargs=kwargs, iterations=10, rounds=100)
