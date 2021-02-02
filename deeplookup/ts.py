from dataclasses import dataclass, field
from typing import NamedTuple
import math

import numpy as np
from numpy.linalg import norm


class N(NamedTuple):
    """A node in the hidden Markov model."""

    ID: str
    prob: float
    klass: float
    ts: int = 0


def from_prob_linear(xx, ts):
    probs = list(zip(xx, ts))
    nodes0 = [N(f"a{i}", prob[0], klass=0, ts=prob[1]) for i, prob in enumerate(probs)]
    nodes1 = [N(f"b{i}", prob[0], klass=1, ts=prob[1]) for i, prob in enumerate(probs)]

    s = N("S", 0.5, 0, 0)
    t0 = N("T0", 0, 0, ts=ts[-1] + 1)
    t1 = N("T1", 1, 1, ts=ts[-1] + 1)

    al = [nodes0, nodes1]
    return al, s, t0, t1


class FIFOQueue:
    def __init__(self):
        self.q = []

    def push(self, item):
        self.q.append(item)

    def pop(self):
        return self.q.pop()

    def empty(self):
        return len(self.q) == 0

    def __contains__(self, key):
        return key in self.q


@dataclass
class Rsamp:
    k: float = field(default=1.35)
    La: float = field(default=1.19)
    _sigma: float = field(default=0.0, init=False)

    def emission_prob(self, n: N) -> float:
        c = 1 / (self._sigma * math.sqrt(2 * math.pi))
        pw = abs(n.klass - n.prob) / self._sigma
        return c * math.exp(-(pw ** 2))

    def transition_prob(self, n1: N, n2: N) -> float:
        kls = 1 - (n1.klass ^ n2.klass)
        delta = abs(n1.ts - n2.ts) / math.exp(self.k * kls)
        return self.La * math.exp(-delta)

    def max_prob(self, adjacency_list, u, v, w):
        prob0 = self.transition_prob(u, v) * self.emission_prob(v)
        prob1 = self.transition_prob(u, w) * self.emission_prob(w)
        return (v, prob0) if prob0 >= prob1 else (w, prob1)

    def search(self, adjacency_list, s, t):
        path = [s]
        probs = []
        joint_prob = self.emission_prob(s)
        records = len(adjacency_list[0])
        _Node = s

        for i in range(records):
            probs.append(joint_prob)
            u = _Node
            v = adjacency_list[0][i]
            w = adjacency_list[1][i]
            _Node, new_prob = self.max_prob(adjacency_list, u, v, w)
            path.append(_Node)
            joint_prob *= new_prob

        _, new_prob = self.max_prob(adjacency_list, _Node, t, t)
        joint_prob *= new_prob

        path.append(t)
        probs.append(joint_prob)

        return joint_prob, path, probs

    def predict(self, x):
        self._sigma = np.sqrt(np.std(x))
        ts = list(range(1, len(x) + 1))

        AL, s, t0, t1 = from_prob_linear(xx=x, ts=ts)
        p0, c0, probs0 = self.search(AL, s, t0)
        p1, c1, probs1 = self.search(AL, s, t1)

        y_pred = [n.klass for n in c0[1:-1]]
        return y_pred
