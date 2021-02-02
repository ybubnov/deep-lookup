from pyts.classification import BOSSVS, SAXVSM

from deeplookup import datasets, metrics
from deeplookup.ts import Rsamp


rsamp = Rsamp()
bossvs = BOSSVS(window_size=36, word_size=12)
saxvsm = SAXVSM(window_size=36, word_size=12)

x_train, y_train = datasets.load_train_ts()
for model in (bossvs, saxvsm):
    model.fit(x_train, y_train)

for name, model, kwargs in [
    ("rsamp", rsamp, {"to_categorical": False}),
    ("bossvs", bossvs, {"to_categorical": True}),
    ("saxvsm", saxvsm, {"to_categorical": True}),
]:
    res = metrics.ts_score(model, **kwargs)
    print(f"metrics for {name}")
    print(res)
    print("-" * 80)
