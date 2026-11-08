"""E8 robustness to misspecified inputs: the nine medium-load core ED types are re-run with one
demand-side input perturbed at a time (capacities unchanged), for every fixed policy and the two
optimized policies of the type. Test replications 0-9, fast KPIs; the unperturbed values for the
same replications come from E2 and E6."""
import os

import pandas as pd

from common import P, pmap, save
from edsim.engine import Engine
from edsim.generator import generate
from edsim.kpis import fast_kpis
from edsim.policies import make_queue_rule, make_routing_rule, policies_for
from opt_common import load_specs, rules_from_spec, spec_for

REPS = list(range(10))
VARIANTS = {
    "phys_minus10": {"phys_mult": 0.9}, "phys_plus10": {"phys_mult": 1.1},
    "diag_minus25": {"diag_mult": 0.75}, "diag_plus25": {"diag_mult": 1.25},
    "cv_half": {"cv_mult": 0.5}, "cv_plus50": {"cv_mult": 1.5},
    "profile_flat": {"profile_power": 0.0}, "profile_peaked": {"profile_power": 1.5},
    "pred_correlated": {"corr_pred": True},
}


def task(args):
    acuity, config, variant, rep = args
    ed = P.build_ed_type(acuity, "medium", config, perturb=VARIANTS[variant])
    pts = generate(ed, rep, alphas=[1.0])
    rows = []
    for q, r in policies_for(config):
        k = fast_kpis(Engine(ed, pts, make_queue_rule(q), make_routing_rule(r) if r else None, alpha=1.0).run())
        rows.append({"acuity": acuity, "config": config, "variant": variant, "rep": rep, "queue": q, "routing": r or "", **k})
    for obj in ("access", "balanced"):
        s = spec_for(load_specs(), acuity, "medium", config, obj)
        if s is None:
            continue
        qq, rr = rules_from_spec(s, "OPT")
        k = fast_kpis(Engine(ed, pts, qq, rr if config != "POOLED" else None, alpha=1.0).run())
        rows.append({"acuity": acuity, "config": config, "variant": variant, "rep": rep,
                     "queue": "OPT-" + obj.upper(), "routing": "OPT-" + obj.upper() + "-R" if config != "POOLED" else "", **k})
    return rows


def main():
    f = P.design()["factors"]
    tasks = [(a, c, v, r) for r in REPS for v in VARIANTS for a in f["acuity"] for c in f["config"]]
    if os.environ.get("EDSIM_QUICK"):     # pilot runs only
        tasks = tasks[:2]
    df = pd.DataFrame([r for rr in pmap(task, tasks) for r in rr])
    save(df, "robustness_kpis", "E8_robustness", reps=REPS, variants=VARIANTS)


if __name__ == "__main__":
    main()
