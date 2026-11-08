"""Simulation-based optimization of similarity policies with the cross-entropy method (CEM).

Decision variables (one optimized policy per ED type and objective):
  queue rule   weights w (6, on the simplex, via softmax), target code for attributes
               B-F (A is fixed at 1, most urgent first), ageing gamma in (0, 0.5),
               reassessment bonus beta in (-0.5, 0.5);
  routing rule (two-area EDs) weights v (6, softmax) and offset b in (-0.5, 0.5), with the
               targets of Proposition 3 (Main 1-1-1-1-1-1, second area 5-5-2-2-4-2).
Continuous variables are sampled from independent normal distributions in an unconstrained
parameterization; target codes from independent categorical distributions. Each iteration
evaluates N candidates on the same training replications (common random numbers), keeps the
best fraction (elite) and moves the sampling distributions toward the elite with smoothing.
Selection uses separate validation replications; the selected policy is then evaluated on
test replications that are never used for search or selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import params as P
from .engine import Engine
from .kpis import fast_kpis
from .policies import Similarity, RouteSimilarity, make_queue_rule, make_routing_rule

N_CLASSES = (5, 5, 2, 2, 4, 2)
TARGET_ATTRS = (1, 2, 3, 4, 5)           # B..F are searched; A fixed at 1
MAIN_TARGET = (1, 1, 1, 1, 1, 1)
SECOND_TARGET = (5, 5, 2, 2, 4, 2)


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


@dataclass
class Candidate:
    z_w: np.ndarray
    z_g: float
    z_r: float
    target: tuple
    z_rw: np.ndarray | None = None
    z_b: float = 0.0

    def queue_spec(self) -> dict:
        w = softmax(self.z_w)
        return {"kind": "similarity", "weights": [float(x) for x in np.round(w, 6)], "target": list(self.target),
                "aging": float(round(0.5 / (1 + math.exp(-self.z_g)), 6)),
                "ra_bonus": float(round(0.5 * math.tanh(self.z_r), 6))}

    def routing_spec(self) -> dict | None:
        if self.z_rw is None:
            return None
        v = softmax(self.z_rw)
        return {"kind": "similarity", "weights": [float(x) for x in np.round(v, 6)],
                "target_main": list(MAIN_TARGET), "target_second": list(SECOND_TARGET), "ctas_floor": 3,
                "offset": float(round(0.5 * math.tanh(self.z_b), 6))}


def build_rules(qspec: dict, rspec: dict | None, name: str):
    q = make_queue_rule(name, qspec)
    r = make_routing_rule(name + "-R", rspec) if rspec else None
    return q, r


def run(ed, pts, qrule, rrule, alpha=1.0):
    return Engine(ed, pts, qrule, rrule if ed.config != "POOLED" else None, alpha=alpha).run()


@dataclass
class CEMSettings:
    iterations: int = 8
    population: int = 20
    elite: int = 5
    smoothing: float = 0.7
    sigma0: float = 1.0
    sigma_floor: float = 0.05
    shortlist: int = 5


@dataclass
class CEMState:
    mu: np.ndarray
    sigma: np.ndarray
    cat: list          # list of probability vectors for B..F
    history: list = field(default_factory=list)


def objective_value(objective: str, k: dict, ref: dict | None) -> float:
    if objective == "access":
        return k["ratp"]
    if objective == "balanced":
        return 0.5 * k["ratp"] / ref["ratp"] + 0.5 * k["los_all_mean"] / ref["los_all_mean"]
    raise ValueError(objective)


class CEMOptimizer:
    """Cross-entropy search for one ED type and one objective."""

    def __init__(self, ed, objective: str, train_lists: list, val_lists: list, settings: CEMSettings,
                 seed: int, init_q: dict, init_r: dict | None, default_rules):
        self.ed = ed
        self.two_area = ed.config != "POOLED"
        self.objective = objective
        self.train = train_lists
        self.val = val_lists
        self.s = settings
        self.rng = np.random.default_rng(seed)
        self.default_rules = default_rules
        self.ref_train = [self._kpis(pts, *default_rules) for pts in train_lists] if objective == "balanced" else [None] * len(train_lists)
        self.ref_val = [self._kpis(pts, *default_rules) for pts in val_lists] if objective == "balanced" else [None] * len(val_lists)
        # initial distribution centred on the given (SIM) policy
        zw = np.log(np.maximum(np.array(init_q["weights"], float), 1e-3))
        g = min(max(init_q.get("aging", 0.0), 1e-3), 0.499)
        zg = math.log(g / (0.5 - g))
        zr = 0.0
        mu = list(zw) + [zg, zr]
        if self.two_area:
            mu += list(np.log(np.maximum(np.array(init_r["weights"], float), 1e-3))) + [0.0]
        self.state = CEMState(np.array(mu), np.full(len(mu), settings.sigma0), [])
        for j, a in enumerate(TARGET_ATTRS):
            p = np.full(N_CLASSES[a], 0.5 / (N_CLASSES[a] - 1))
            p[init_q["target"][a] - 1] = 0.5
            self.state.cat.append(p)
        self.init_candidate = self._decode(self.state.mu, tuple(init_q["target"]))
        self.archive = []      # (J_train, Candidate, kpis)
        self.evaluations = 0

    # -------------------------------------------------------------- helpers
    def _decode(self, x, target):
        z_w, z_g, z_r = x[:6], x[6], x[7]
        if self.two_area:
            return Candidate(np.array(z_w), float(z_g), float(z_r), target, np.array(x[8:14]), float(x[14]))
        return Candidate(np.array(z_w), float(z_g), float(z_r), target)

    def _kpis(self, pts, q, r):
        self.evaluations = getattr(self, "evaluations", 0) + 1
        return fast_kpis(run(self.ed, pts, q, r))

    def evaluate(self, cand: Candidate, lists, refs) -> tuple:
        q, r = build_rules(cand.queue_spec(), cand.routing_spec(), "OPT")
        ks = [self._kpis(pts, q, r) for pts in lists]
        js = [objective_value(self.objective, k, ref) for k, ref in zip(ks, refs)]
        agg = {k: float(np.mean([x[k] for x in ks])) for k in ("ratp", "awt", "los_all_mean")}
        return float(np.mean(js)), agg

    def sample(self) -> Candidate:
        x = self.state.mu + self.state.sigma * self.rng.standard_normal(len(self.state.mu))
        target = [1] + [0] * 5
        for j, a in enumerate(TARGET_ATTRS):
            target[a] = int(self.rng.choice(N_CLASSES[a], p=self.state.cat[j])) + 1
        return self._decode(x, tuple(target))

    @staticmethod
    def _vector(c: Candidate, two_area: bool):
        v = list(c.z_w) + [c.z_g, c.z_r]
        if two_area:
            v += list(c.z_rw) + [c.z_b]
        return np.array(v)

    # -------------------------------------------------------------- search
    def search(self):
        s = self.s
        for it in range(s.iterations):
            cands = [self.init_candidate] if it == 0 else []
            while len(cands) < s.population:
                cands.append(self.sample())
            scored = []
            for c in cands:
                j, agg = self.evaluate(c, self.train, self.ref_train)
                scored.append((j, c, agg))
                self.archive.append((j, c, agg))
            scored.sort(key=lambda t: t[0])
            elite = scored[: s.elite]
            ev = np.array([self._vector(c, self.two_area) for _, c, _ in elite])
            a = s.smoothing
            self.state.mu = a * ev.mean(axis=0) + (1 - a) * self.state.mu
            self.state.sigma = np.maximum(a * ev.std(axis=0) + (1 - a) * self.state.sigma, s.sigma_floor)
            for j, at in enumerate(TARGET_ATTRS):
                freq = np.bincount([c.target[at] - 1 for _, c, _ in elite], minlength=N_CLASSES[at]) / len(elite)
                self.state.cat[j] = a * freq + (1 - a) * self.state.cat[j]
                self.state.cat[j] /= self.state.cat[j].sum()
            self.state.history.append({"iteration": it, "best_J": scored[0][0], "elite_mean_J": float(np.mean([x[0] for x in elite])),
                                       "population_mean_J": float(np.mean([x[0] for x in scored])),
                                       "sigma_mean": float(self.state.sigma.mean())})
        # selection on validation replications among the best distinct training candidates
        self.archive.sort(key=lambda t: t[0])
        short, seen = [], set()
        for j, c, agg in self.archive:
            key = (tuple(np.round(self._vector(c, self.two_area), 3)), c.target)
            if key in seen:
                continue
            seen.add(key)
            short.append((j, c))
            if len(short) == s.shortlist:
                break
        val = [(self.evaluate(c, self.val, self.ref_val)[0], jt, c) for jt, c in short]
        val.sort(key=lambda t: t[0])
        best_val, best_train, best = val[0]
        return best, {"J_train": best_train, "J_validation": best_val, "shortlist_validation": [v[0] for v in val],
                      "evaluations": self.evaluations}
