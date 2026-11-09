"""Verification tests: queue rules, similarity propositions, KPI definitions,
patient conservation, time-stamp order, determinism and confusion matrices."""
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from edsim import params as P
from edsim.engine import Engine, RuleQueue
from edsim.generator import (CODE_RANGES, Patient, confusion_at, confusion_matrices,
                             generate, reference_confusion)
from edsim.kpis import awt, compute_kpis, door_to_physician, ptp_by_ctas, ratp
from edsim.policies import (APQ, FCFS, CTASPriority, CTASThen, RouteCTAS, RoutePred,
                            RouteSimilarity, Similarity, make_queue_rule, similarity)
from edsim.runner import run_policy

ALL_CODES = list(itertools.product(range(1, 6), range(1, 6), (1, 2), (1, 2), range(1, 5), (1, 2)))


# ------------------------------------------------------------------ queue rules
def _fill(rule, items, now):
    q = RuleQueue(rule)
    for pid, (ctas, code, t) in enumerate(items):
        q.push(pid, ctas, code, t, pid, None)
    order = []
    while q.n:
        order.append(q.pop(now)[0])
    return order


C = (3, 3, 2, 2, 4, 2)   # neutral code


def test_fcfs_with_ctas1_first():
    items = [(3, C, 0.0), (5, C, 1.0), (1, C, 5.0), (2, C, 2.0)]
    assert _fill(FCFS(), items, 10.0) == [2, 0, 1, 3]


def test_ctas_priority_fcfs_within_level():
    items = [(3, C, 0.0), (2, C, 3.0), (3, C, 1.0), (2, C, 2.0), (5, C, -10.0)]
    assert _fill(CTASPriority(), items, 10.0) == [3, 1, 0, 2, 4]


def test_ctas_then_short_service():
    short = (3, 5, 2, 2, 4, 2)
    long_ = (3, 1, 2, 2, 4, 2)
    items = [(3, long_, 0.0), (3, short, 5.0), (2, long_, 9.0)]
    assert _fill(CTASThen("x", "short_service"), items, 10.0) == [2, 1, 0]


def test_apq_serves_largest_fraction_of_target():
    # at t=100: CTAS3 waited 50/30=1.67 targets; CTAS4 waited 100/60=1.67 (tie -> earlier t_enter); CTAS5 90/120
    items = [(3, C, 50.0), (4, C, 0.0), (5, C, 10.0)]
    assert _fill(APQ(), items, 100.0)[0] == 1
    items = [(3, C, 40.0), (4, C, 0.0)]       # CTAS3 2.0 targets beats CTAS4 1.67
    assert _fill(APQ(), items, 100.0)[0] == 0


@pytest.mark.parametrize("rule", [Similarity("s", [0.4, 0.3, 0.1, 0.1, 0.05, 0.05], [1, 5, 2, 2, 4, 2], aging=0.1),
                                  APQ()])
def test_dynamic_rules_heap_equals_full_scan(rule):
    """The per-CTAS heaps give the same first choice as scanning all patients by
    effective priority (score + ageing, or wait/target), at several times."""
    rng = np.random.default_rng(7)
    ub = P.v(P.parameters()["ctas_targets_minutes"])
    idx = rng.choice(len(ALL_CODES), 60, replace=False)
    items = [(ALL_CODES[i][0] if ALL_CODES[i][0] > 1 else 2, ALL_CODES[i], float(rng.uniform(0, 300))) for i in idx]
    for now in (300.0, 600.0, 3000.0):
        def eff(k):
            c, code, t = items[k]
            if isinstance(rule, APQ):
                return (-(now - t) / ub[c - 1], t, k)
            return (-(rule.score(code) + 0.1 * (now - t) / ub[c - 1]), t, k)
        brute = min(range(len(items)), key=eff)
        assert _fill(rule, items, now)[0] == brute


def test_ageing_lets_long_waiting_low_acuity_patient_overtake():
    r = Similarity("s", [0.4, 0.3, 0.1, 0.1, 0.05, 0.05], [1, 5, 2, 2, 4, 2], aging=0.1)
    items = [(4, (4, 1, 1, 1, 1, 1), 0.0), (3, (3, 5, 2, 2, 4, 2), 995.0)]
    assert _fill(r, items, 1000.0)[0] == 0      # 0.1 + 0.1*1000/60 = 1.77 > 0.8 + 0.017
    assert _fill(Similarity("s", r.weights, r.target), items, 1000.0)[0] == 1


# ------------------------------------------------------------------ propositions
def _order(rule_key, codes):
    return sorted(range(len(codes)), key=lambda i: rule_key(codes[i]) + (i,))


@pytest.mark.parametrize("wa,wb,expect_exact", [(0.85, 0.15, True), (0.25, 0.25, False), (0.40, 0.30, False)])
def test_proposition_lexicographic_condition(wa, wb, expect_exact):
    """Proposition 2: w_A/R_A > w_B * D_B/R_B gives exactly CTAS-then-short-service."""
    sim = Similarity("s", [wa, wb, 0, 0, 0, 0], [1, 5, 2, 2, 4, 2])
    ref = lambda c: (c[0], -c[1])
    s = lambda c: (-sim.score(c),)
    codes = sorted({(a, b, 2, 2, 4, 2) for a in range(1, 6) for b in range(1, 6)})
    exact = all((ref(x) < ref(y)) == (s(x) < s(y)) for x in codes for y in codes if ref(x) != ref(y))
    assert exact == expect_exact
    assert (wa / CODE_RANGES[0] > wb) == expect_exact


def test_proposition_strict_ctas():
    """Only w_A > 0: similarity order equals CTAS order (ties FCFS)."""
    sim = make_queue_rule("SIMREP-CTAS")
    ref = make_queue_rule("CTAS")
    items = [(c[0], c, float(t)) for t, c in enumerate(ALL_CODES[::7])]
    assert _fill(sim, items, 1e6) == _fill(ref, items, 1e6)


@pytest.mark.parametrize("rep,refname", [("SIMREP-CTAS-SPT", "CTAS-SPT"), ("SIMREP-CTAS-PD", "CTAS-PD")])
def test_proposition_exact_representations(rep, refname):
    sim, ref = make_queue_rule(rep), make_queue_rule(refname)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(ALL_CODES), 300, replace=False)
    items = [(ALL_CODES[i][0], ALL_CODES[i], float(rng.integers(0, 50))) for i in idx]
    assert _fill(sim, items, 100.0) == _fill(ref, items, 100.0)


def test_proposition_routing_ctas_representation():
    """Proposition 3: two-target routing with weight on CTAS only equals CTAS streaming."""
    rr = RouteSimilarity("x", [1, 0, 0, 0, 0, 0], [1] * 6, [5, 5, 2, 2, 4, 2], ctas_floor=1)
    ref = RouteCTAS()
    assert all(rr.to_second(c[0], c) == ref.to_second(c[0], c) for c in ALL_CODES)


@pytest.mark.parametrize("wb,wc", [(0.4, 0.3), (0.2, 0.5), (0.6, 0.2), (0.3, 0.3)])
def test_proposition_routing_offset_reproduces_prediction_streaming(wb, wc):
    """Proposition 4: weights on B and C only, w_B < 4 w_C and an offset in
    [max(w_C, w_B - w_C), w_C + w_B/2) reproduce prediction streaming on every code."""
    lo, hi = round(max(wc, wb - wc), 9), round(wc + wb / 2, 9)   # scores are compared after rounding to 1e-9
    for b in (lo, 0.5 * (lo + hi), hi - 1e-6):
        rr = RouteSimilarity("x", [0, wb, wc, 0, 0, 0], [1] * 6, [5, 5, 2, 2, 4, 2], ctas_floor=3, offset=b)
        assert all(rr.to_second(c[0], c) == RoutePred().to_second(c[0], c) for c in ALL_CODES), b
    # outside the interval the representation fails
    for b in (lo - 0.01, hi + 0.01):
        rr = RouteSimilarity("x", [0, wb, wc, 0, 0, 0], [1] * 6, [5, 5, 2, 2, 4, 2], ctas_floor=3, offset=b)
        assert any(rr.to_second(c[0], c) != RoutePred().to_second(c[0], c) for c in ALL_CODES), b


def test_prediction_streaming_not_representable_by_default_similarity():
    rr = RouteSimilarity("x", [0.4, 0.3, 0.1, 0.1, 0.05, 0.05], [1] * 6, [5, 5, 2, 2, 4, 2])
    ref = RoutePred()
    disagree = sum(rr.to_second(c[0], c) != ref.to_second(c[0], c) for c in ALL_CODES)
    assert disagree > 0


def test_similarity_rounding_makes_equal_scores_tie():
    w = [0.1] * 6
    a = similarity((2, 3, 1, 2, 2, 1), (1, 5, 2, 2, 4, 2), w)
    b = similarity((2, 3, 1, 2, 2, 1), (1, 5, 2, 2, 4, 2), w)
    assert a == b and len(str(a).split(".")[-1]) <= 9


# ------------------------------------------------------------------ confusion matrices
def test_confusion_rows_and_alpha():
    for name, m in confusion_matrices().items():
        assert np.allclose(m.sum(axis=1), 1)
        assert np.allclose(confusion_at(0.0, m), np.eye(m.shape[0]))
        assert np.allclose(confusion_at(1.0, m), m)
        assert np.allclose(confusion_at(1.5, m).sum(axis=1), 1)
    ref = reference_confusion(5, 0.45, 0.35)
    assert ref[0, 1] > ref[0, 2] > ref[0, 3]


# ------------------------------------------------------------------ KPI definitions on a hand-computed example
def test_kpis_hand_computed():
    # targets 15,15,30,60,120; door-to-physician waits:
    pt = pd.DataFrame({"ctas": [1, 2, 3, 3, 4, 5], "arrival": [0.0] * 6,
                       "t_pia": [10.0, 20.0, 20.0, 50.0, 90.0, 100.0]})
    assert list(door_to_physician(pt)) == [10, 20, 20, 50, 90, 100]
    ptp = ptp_by_ctas(pt)
    assert ptp == {1: 0.0, 2: 1.0, 3: 0.5, 4: 1.0, 5: 0.0}
    assert math.isclose(ratp(pt), 0 + 1 / 2 + 0.5 / 3 + 1 / 4 + 0)
    # lateness: CTAS2 5 min; CTAS3 mean(0,20)=10; CTAS4 30
    assert math.isclose(awt(pt), 5 / 2 + 10 / 3 + 30 / 4)


# ------------------------------------------------------------------ engine checks
@pytest.fixture(scope="module")
def small_runs():
    out = {}
    for cfg, rt in (("POOLED", None), ("FASTTRACK", "STREAM-CTAS"), ("SPLITFLOW", "STREAM-PRED")):
        ed = P.build_ed_type("community", "medium", cfg)
        pts = generate(ed, 0)
        out[cfg] = (ed, pts, run_policy(ed, pts, "SIM-AGE", rt))
    return out


def test_patient_conservation_and_order(small_runs):
    for cfg, (ed, pts, eng) in small_runs.items():
        log = pd.DataFrame(eng.log, columns=["pid", "stage", "resource", "t_req", "t_start", "t_end"])
        obs = [p.pid for p in pts if eng.warm <= p.arrival < eng.obs_end]
        done = set(log[log.stage.isin(["DD", "BOARD"])].pid)
        assert set(obs) <= done, cfg                      # every observed patient left
        assert (log.t_req <= log.t_start + 1e-9).all()
        assert (log.t_start <= log.t_end + 1e-9).all()
        arr = {p.pid: p.arrival for p in pts}
        assert (log.t_req >= log.pid.map(arr) - 1e-9).all()
        # each departed patient has exactly one AS and one decision, and one DD or BOARD
        d = log[log.pid.isin(obs)]
        assert (d[d.stage == "AS"].groupby("pid").size() == 1).all()
        assert (d[d.stage == "DECISION"].groupby("pid").size() == 1).all()
        assert (d[d.stage.isin(["DD", "BOARD"])].groupby("pid").size() == 1).all()
        # sequence: triage <= first space <= AS start <= decision <= departure
        first = d.groupby(["pid", "stage"]).t_start.min().unstack()
        sp = d[d.stage == "SPACE"].groupby("pid").t_start.min()
        assert (first.TRIAGE <= sp + 1e-9).all() and (sp <= first.AS + 1e-9).all()
        assert (first.AS <= first.DECISION + 1e-9).all()
        # pathway steps all present
        for p in pts[:300]:
            if p.pid in obs:
                st = list(d[(d.pid == p.pid) & d.stage.isin(P.DIAG_STAGES)].sort_values("t_start").stage)
                assert tuple(st) == p.steps


def test_capacity_never_exceeded(small_runs):
    for cfg, (ed, pts, eng) in small_runs.items():
        log = pd.DataFrame(eng.log, columns=["pid", "stage", "resource", "t_req", "t_start", "t_end"])
        for res in list(eng.spaces) + list(P.DIAG_STAGES):
            r = log[(log.stage == "SPACE") & (log.resource == res)] if res in eng.spaces else log[log.stage == res]
            ev = sorted([(t, 1) for t in r.t_start] + [(t, -1) for t in r.t_end], key=lambda x: (x[0], x[1]))
            cur = mx = 0
            for _, d in ev:
                cur += d
                mx = max(mx, cur)
            cap = eng.spaces[res].cap if res in eng.spaces else eng.diag[res].cap
            assert mx <= cap, (cfg, res, mx, cap)


def test_determinism_same_policy_same_seed():
    ed = P.build_ed_type("tertiary", "high", "FASTTRACK")
    a = compute_kpis(run_policy(ed, generate(ed, 3), "APQ", "STREAM-SIM"))
    b = compute_kpis(run_policy(ed, generate(ed, 3), "APQ", "STREAM-SIM"))
    assert a == b


def test_overall_mean_between_group_means(small_runs):
    for cfg, (ed, pts, eng) in small_runs.items():
        k = compute_kpis(eng)
        g = [k[f"los_ctas{l}_mean"] for l in range(1, 6) if not math.isnan(k[f"los_ctas{l}_mean"])]
        assert min(g) <= k["los_all_mean"] <= max(g)
        assert min(k["los_disch_mean"], k["los_adm_mean"]) <= k["los_all_mean"] <= max(k["los_disch_mean"], k["los_adm_mean"])


def test_generator_crn_across_configs():
    a = generate(P.build_ed_type("community", "medium", "POOLED"), 2)
    b = generate(P.build_ed_type("community", "medium", "SPLITFLOW"), 2)
    assert [p.arrival for p in a] == [p.arrival for p in b]
    assert [p.pred[1.0] for p in a] == [p.pred[1.0] for p in b]


def test_alpha_zero_is_truth():
    pts = generate(P.build_ed_type("urgentcare", "low", "POOLED"), 0)
    assert all(p.pred[0.0] == p.code_true for p in pts)
    acc = np.mean([p.pred[1.0][2] == p.code_true[2] for p in pts])
    assert 0.80 < acc < 0.90


# ------------------------------------------------------------------ v1.1 extensions
def test_fast_kpis_equal_full_kpis(small_runs):
    from edsim.kpis import fast_kpis
    for cfg, (ed, pts, eng) in small_runs.items():
        a, b = fast_kpis(eng), compute_kpis(eng)
        for k in ("ratp", "awt", "los_all_mean", "pia_all_mean", "n_patients"):
            assert math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-12), (cfg, k)


def test_named_and_numeric_ed_types_identical():
    """A type built from numeric inputs equal to a named type gives the same patients."""
    a = P.build_ed_type("tertiary", "medium", "FASTTRACK", boarding="high", diagnostics="heavy")
    b = P.build_ed_type(P.mix_value("tertiary"), P.v(P.parameters()["load_levels"])["medium"], "FASTTRACK",
                        boarding=P.boarding_mean_value("high"), diagnostics=P.diag_shift_value("heavy"),
                        seed_parts=a.seed_parts)
    assert (a.physicians_by_shift, a.spaces_total, a.visits_per_day) == (b.physicians_by_shift, b.spaces_total, b.visits_per_day)
    pa, pb = generate(a, 1), generate(b, 1)
    assert [(p.arrival, p.ctas, p.d_end, p.pred[1.0]) for p in pa] == [(p.arrival, p.ctas, p.d_end, p.pred[1.0]) for p in pb]


def test_ra_bonus_and_routing_offset():
    r = Similarity("s", [1, 0, 0, 0, 0, 0], [1, 5, 2, 2, 4, 2], ra_bonus=0.3)
    # CTAS3 reassessment (score 0.5 + 0.3) beats CTAS2 new patient (0.75)
    q = RuleQueue(r)
    q.push(0, 2, (2, 3, 2, 2, 4, 2), 0.0, 0, "AS")
    q.push(1, 3, (3, 3, 2, 2, 4, 2), 5.0, 1, "RA")
    assert q.pop(10.0)[0] == 1
    rr0 = RouteSimilarity("x", [1, 0, 0, 0, 0, 0], [1] * 6, [5, 5, 2, 2, 4, 2], ctas_floor=1)
    rr1 = RouteSimilarity("x", [1, 0, 0, 0, 0, 0], [1] * 6, [5, 5, 2, 2, 4, 2], ctas_floor=1, offset=-0.3)
    c3 = (3, 3, 2, 2, 4, 2)
    assert not rr0.to_second(3, c3) and rr1.to_second(3, c3)


def test_perturbations_change_demand_not_capacity():
    a = P.build_ed_type("community", "medium", "POOLED")
    b = P.build_ed_type("community", "medium", "POOLED", perturb={"diag_mult": 1.25, "corr_pred": True})
    assert (a.spaces_total, a.physicians_by_shift, a.visits_per_day) == (b.spaces_total, b.physicians_by_shift, b.visits_per_day)
    pa, pb = generate(a, 0), generate(b, 0)
    assert [p.arrival for p in pa] == [p.arrival for p in pb]
    lb = [(x.d_steps, y.d_steps) for x, y in zip(pa, pb) if "LB" in x.steps][:50]
    assert all(math.isclose(y[0] / x[0], 1.25) for x, y in lb if x)
    assert all(len(set(p.pred[1.0][1:]) ) >= 1 for p in pb)
