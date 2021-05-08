import pathlib

import pytest

from deeplookup.experiments import tunneling


kwargs = {"limit": 1, "prepare_skip": True, "prepare_predict": True}


def model_ids(models):
    return [pathlib.Path(model[3]).stem for model in models]


@pytest.mark.parametrize(
    "train, evaluate, factory, path, kw, layers",
    tunneling.B_MODELS,
    ids=model_ids(tunneling.B_MODELS),
)
def test_binary_benchmark(train, evaluate, factory, path, kw, layers, benchmark):
    function = tunneling.fit(train, evaluate, factory, path, kw, layers)
    benchmark.pedantic(function, kwargs=kwargs, iterations=10, rounds=100)


@pytest.mark.parametrize(
    "train, evaluate, factory, path, kw",
    tunneling.M_MODELS,
    ids=model_ids(tunneling.M_MODELS),
)
def test_multilabel_benchmark(train, evaluate, factory, path, kw, benchmark):
    function = tunneling.fit_multilabel(train, evaluate, factory, path, kw)
    benchmark.pedantic(function, kwargs=kwargs, iterations=10, rounds=100)
