"""E5 sensitivity: boarding (high) and diagnostic intensity (light, heavy), one at a time,
on the nine medium-load core ED types, all policies, alpha = 1.
Capacities stay at their core sizing, so these factors change demand only.
"""
import pandas as pd

from common import P, pmap, save
from edsim.policies import policies_for
from edsim.runner import run_ed_rep

VARIANTS = [("high", "typical"), ("low", "light"), ("low", "heavy")]


def task(args):
    acuity, config, boarding, diagnostics, rep = args
    ed = P.build_ed_type(acuity, "medium", config, boarding=boarding, diagnostics=diagnostics)
    alpha = P.v(P.parameters()["prediction"]["alpha_main"])
    rows = run_ed_rep(ed, rep, policies_for(config), alpha=alpha, alphas=[alpha])
    for r in rows:
        r["visits_per_day"] = ed.visits_per_day
    return rows


def main():
    reps = P.v(P.parameters()["run_control"]["replications_secondary"])
    f = P.design()["factors"]
    tasks = [(a, c, b, d, r) for r in range(reps) for a in f["acuity"] for c in f["config"] for b, d in VARIANTS]
    df = pd.DataFrame([row for chunk in pmap(task, tasks) for row in chunk])
    save(df, "kpis_sensitivity", "E5_sensitivity", replications=reps, load="medium", variants=VARIANTS)


if __name__ == "__main__":
    main()
