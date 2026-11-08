"""E9 out-of-sample ED types: 30 ED types drawn by Latin hypercube sampling over continuous
factors not used to build the decision guide: acuity position t in [0, 1] (urgent-care-like at 0,
community at 0.5, tertiary at 1; CTAS shares interpolated linearly), offered load in [0.70, 0.90],
mean boarding time in [120, 360] minutes and diagnostic shift in [-0.3, 0.3]; ten ED types per
configuration. Every fixed policy and the optimized policies of the nearest core ED type are run
(replications 0-9, fast KPIs)."""
import os

import numpy as np
import pandas as pd

from common import P, pmap, save
from edsim.engine import Engine
from edsim.generator import generate
from edsim.kpis import fast_kpis
from edsim.policies import make_queue_rule, make_routing_rule, policies_for
from opt_common import load_specs, rules_from_spec, spec_for

N_PER_CONFIG = 10
REPS = list(range(10))
SEED = 20261109


def design_points():
    rng = np.random.default_rng(SEED)
    pts = []
    for ci, config in enumerate(P.design()["factors"]["config"]):
        n = N_PER_CONFIG
        u = (np.array([rng.permutation(n) for _ in range(4)]).T + rng.random((n, 4))) / n   # LHS in [0,1]^4
        for i in range(n):
            t, ld, bd, dg = u[i]
            pts.append({"id": ci * n + i, "config": config, "acuity_t": float(t), "load_value": 0.70 + 0.20 * float(ld),
                        "board_mean": 120 + 240 * float(bd), "diag_shift": -0.3 + 0.6 * float(dg)})
    return pts


def mix_at(t):
    u, c, te = (np.array(P.mix_value(x)) for x in ("urgentcare", "community", "tertiary"))
    m = u + (c - u) * (t / 0.5) if t <= 0.5 else c + (te - c) * ((t - 0.5) / 0.5)
    return list(m / m.sum())


def nearest_core(dp):
    t = dp["acuity_t"]
    acuity = "urgentcare" if t < 0.25 else ("community" if t < 0.75 else "tertiary")
    lv = P.v(P.parameters()["load_levels"])
    load = min(lv, key=lambda k: abs(lv[k] - dp["load_value"]))
    return acuity, load


def task(args):
    dp, rep = args
    ed = P.build_ed_type(mix_at(dp["acuity_t"]), dp["load_value"], dp["config"], boarding=dp["board_mean"],
                         diagnostics=dp["diag_shift"], seed_parts=(90, dp["id"], 0, 0), label=f"oos{dp['id']}")
    pts = generate(ed, rep, alphas=[1.0])
    base = {"id": dp["id"], **{k: v for k, v in dp.items() if k != "id"}, "rep": rep,
            "near_acuity": nearest_core(dp)[0], "near_load": nearest_core(dp)[1], "visits_per_day": ed.visits_per_day}
    rows = []
    for q, r in policies_for(dp["config"]):
        k = fast_kpis(Engine(ed, pts, make_queue_rule(q), make_routing_rule(r) if r else None, alpha=1.0).run())
        rows.append({**base, "queue": q, "routing": r or "", **k})
    a, l = nearest_core(dp)
    for obj in ("access", "balanced"):
        s = spec_for(load_specs(), a, l, dp["config"], obj)
        if s is None:
            continue
        qq, rr = rules_from_spec(s, "OPT")
        k = fast_kpis(Engine(ed, pts, qq, rr if dp["config"] != "POOLED" else None, alpha=1.0).run())
        rows.append({**base, "queue": "OPT-" + obj.upper(), "routing": "OPT-" + obj.upper() + "-R" if dp["config"] != "POOLED" else "", **k})
    return rows


def main():
    tasks = [(dp, r) for r in REPS for dp in design_points()]
    if os.environ.get("EDSIM_QUICK"):     # pilot runs only
        tasks = tasks[:2]
    df = pd.DataFrame([r for rr in pmap(task, tasks) for r in rr])
    save(df, "oos_kpis", "E9_out_of_sample", reps=REPS, seed=SEED, n_per_config=N_PER_CONFIG)


if __name__ == "__main__":
    main()
