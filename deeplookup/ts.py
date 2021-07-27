import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class N(NamedTuple):
    """A node in the hidden Markov model."""

    ID: str
    prob: float
    klass: float
    ts: int = 0


@dataclass
class Rsamp:
    k: float = field(default=1.35)
    La: float = field(default=1.19)
    _sigma: float = field(default=0.0, init=False)
    _window: float = field(default=0.1, init=False)

    def emission_prob(self, n: N) -> float:
        c = 1 / (self._sigma * math.sqrt(2 * math.pi))
        pw = abs(n.klass - n.prob) / self._sigma
        return c * math.exp(-(pw ** 2))

    def transition_prob(self, n1: N, n2: N) -> float:
        kls = 1 - (n1.klass ^ n2.klass)
        delta = abs(n1.ts - n2.ts) / math.exp(self.k * kls)
        return self.La * math.exp(-delta)

    def max_prob(self, u, v, w):
        prob0 = self.transition_prob(u, v) * self.emission_prob(v)
        prob1 = self.transition_prob(u, w) * self.emission_prob(w)
        return (v, prob0) if prob0 >= prob1 else (w, prob1)

    def search(self, x, s):
        joint_prob = self.emission_prob(s)
        records = len(x)
        y_pred = np.zeros(records)
        _Node = s

        for i in range(records):
            u = _Node
            v = N("a", x[i], klass=0, ts=i + 1)
            w = N("b", x[i], klass=1, ts=i + 1)
            _Node, new_prob = self.max_prob(u, v, w)
            y_pred[i] = _Node.klass
            joint_prob *= new_prob

        return y_pred

    def predict_proba(self, xx):
        num_samples = len(xx)
        res = np.zeros(shape=(num_samples, 2))
        for i in range(num_samples):
            pred = self.predict(xx[i])
            proba = (np.mean(pred) / self._window) % 1.0
            res[i][0], res[i][1] = 1 - proba, proba
        return res

    def predict(self, x):
        self._sigma = np.sqrt(np.std(x))
        s = N("S", 0.5, 0, 0)
        return self.search(x, s)
