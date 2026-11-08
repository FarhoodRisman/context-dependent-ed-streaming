"""Queue rules and routing rules.

A queue rule orders the patients waiting for one resource. It provides
  static_key(code, ctas, t_enter, seq): order inside one CTAS level (never changes)
  dyn_key(static_key, ctas, now): key used to compare the heads of the CTAS levels.
Inside one CTAS level the two orders always agree, so each queue keeps one heap
per CTAS level and only compares at most five heads at dispatch time.
Smaller keys are served first. CTAS 1 is served first by the queue itself
(safety rule), before any rule key is consulted. Ties end in (t_enter, seq):
first-come-first-served, then patient id.
"""
from __future__ import annotations

from . import params as P
from .generator import CODE_RANGES

ROUND = 9   # similarity scores are rounded to 1e-9 before comparison (explicit ties)


def similarity(code, target, weights) -> float:
    """Weighted Offodile similarity between a patient code and a target code."""
    s = 0.0
    for p, q, w, r in zip(code, target, weights, CODE_RANGES):
        if w:
            s += w * (1.0 - abs(p - q) / r)
    return round(s, ROUND)


def _targets():
    return P.v(P.parameters()["ctas_targets_minutes"])


class QueueRule:
    name = "base"
    dynamic = False

    def static_key(self, code, ctas, t_enter, seq, task=None):
        raise NotImplementedError

    def dyn_key(self, skey, ctas, now):
        return skey


class FCFS(QueueRule):
    def __init__(self, name="FCFS"):
        self.name = name

    def static_key(self, code, ctas, t_enter, seq, task=None):
        return (t_enter, seq)


class CTASPriority(QueueRule):
    def __init__(self, name="CTAS"):
        self.name = name

    def static_key(self, code, ctas, t_enter, seq, task=None):
        return (t_enter, seq)

    def dyn_key(self, skey, ctas, now):
        return (ctas,) + skey


class CTASThen(QueueRule):
    """CTAS first, then a secondary predicted attribute."""

    def __init__(self, name, secondary):
        self.name = name
        self.secondary = secondary

    def static_key(self, code, ctas, t_enter, seq, task=None):
        if self.secondary == "short_service":
            k = -code[1]            # class 5 = very short is served first
        elif self.secondary == "discharge":
            k = -code[2]            # 2 = predicted discharge first
        elif self.secondary == "admission":
            k = code[2]             # 1 = predicted admission first
        else:
            raise ValueError(self.secondary)
        return (k, t_enter, seq)

    def dyn_key(self, skey, ctas, now):
        return (ctas,) + skey


class APQ(QueueRule):
    """Accumulating priority: priority = wait / CTAS target; largest served first."""
    dynamic = True

    def __init__(self, name="APQ"):
        self.name = name
        self.ub = _targets()

    def static_key(self, code, ctas, t_enter, seq, task=None):
        return (t_enter, seq)

    def dyn_key(self, skey, ctas, now):
        return (-(now - skey[0]) / self.ub[ctas - 1],) + skey


class Similarity(QueueRule):
    """Higher similarity to the target code is served first; optional ageing term
    adds `aging` per elapsed CTAS target time; optional `ra_bonus` is added to the score
    of patients waiting for a physician reassessment (in-progress patients)."""

    def __init__(self, name, weights, target, aging=0.0, ra_bonus=0.0):
        self.name = name
        self.weights = tuple(weights)
        self.target = tuple(target)
        self.aging = aging
        self.ra_bonus = ra_bonus
        self.dynamic = aging > 0
        self.ub = _targets()

    def score(self, code):
        return similarity(code, self.target, self.weights)

    def static_key(self, code, ctas, t_enter, seq, task=None):
        s = self.score(code)
        if self.ra_bonus and task == "RA":
            s = round(s + self.ra_bonus, ROUND)
        if self.aging:
            s = round(s - self.aging * t_enter / self.ub[ctas - 1], ROUND)
        return (-s, t_enter, seq)

    def dyn_key(self, skey, ctas, now):
        if self.aging:
            return (skey[0] - self.aging * now / self.ub[ctas - 1],) + skey[1:]
        return skey


def make_queue_rule(name: str, spec: dict | None = None) -> QueueRule:
    cfg = P.policies_config()
    if spec is None:
        spec = cfg["queue_rules"].get(name) or cfg["similarity_representations"][name]
    kind = spec["kind"]
    if kind == "fcfs":
        return FCFS(name)
    if kind == "ctas":
        return CTASPriority(name)
    if kind == "apq":
        return APQ(name)
    if kind == "ctas_then":
        return CTASThen(name, spec["secondary"])
    if kind == "similarity":
        return Similarity(name, spec["weights"], spec["target"], spec.get("aging", 0.0), spec.get("ra_bonus", 0.0))
    raise ValueError(kind)


# ---------------------------------------------------------------- routing
class RoutingRule:
    """Returns True if the patient goes to the second area (fast track / vertical)."""
    name = "base"

    def to_second(self, ctas, code) -> bool:
        raise NotImplementedError


class RouteCTAS(RoutingRule):
    def __init__(self, name="STREAM-CTAS"):
        self.name = name

    def to_second(self, ctas, code):
        return ctas >= 4


class RoutePred(RoutingRule):
    def __init__(self, name="STREAM-PRED"):
        self.name = name

    def to_second(self, ctas, code):
        return ctas >= 3 and code[2] == 2 and code[1] >= 4


class RouteSimilarity(RoutingRule):
    """Second area if S_second - S_main > offset (offset 0: Proposition 3; a non-zero offset
    moves the threshold away from the centre of the code ranges)."""

    def __init__(self, name, weights, target_main, target_second, ctas_floor=3, offset=0.0):
        self.name = name
        self.weights = tuple(weights)
        self.tm = tuple(target_main)
        self.ts = tuple(target_second)
        self.floor = ctas_floor
        self.offset = offset

    def to_second(self, ctas, code):
        if ctas < self.floor:
            return False
        d = round(similarity(code, self.ts, self.weights) - similarity(code, self.tm, self.weights), ROUND)
        return d > self.offset


def make_routing_rule(name: str, spec: dict | None = None) -> RoutingRule:
    cfg = P.policies_config()
    if spec is None:
        spec = cfg["routing_rules"].get(name) or cfg["similarity_representations"][name]
    kind = spec["kind"]
    if kind == "ctas":
        return RouteCTAS(name)
    if kind == "pred":
        return RoutePred(name)
    if kind == "similarity":
        return RouteSimilarity(name, spec["weights"], spec["target_main"], spec["target_second"],
                               spec.get("ctas_floor", 3), spec.get("offset", 0.0))
    raise ValueError(kind)


def queue_rule_names() -> list:
    return list(P.policies_config()["queue_rules"].keys())


def routing_rule_names() -> list:
    return list(P.policies_config()["routing_rules"].keys())


def policy_name(queue: str, routing: str | None) -> str:
    return queue if routing is None else f"{queue}|{routing}"


def policies_for(config: str) -> list:
    """All (queue rule, routing rule) pairs evaluated for a configuration."""
    if config == "POOLED":
        return [(q, None) for q in queue_rule_names()]
    return [(q, r) for q in queue_rule_names() for r in routing_rule_names()]
