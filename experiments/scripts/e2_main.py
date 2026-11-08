"""E2 main experiment: 27 core ED types x all policies, alpha = 1 (reference prediction accuracy).

Every replication generates one patient list per ED type; all policies run on it
(common random numbers). Output: one row of KPIs per (ED type, policy, replication).
"""
import sys

import pandas as pd

from common import P, pmap, save
from edsim.policies import policies_for
from edsim.runner import run_ed_rep


def task(args):
    acuity, load, config, rep = args
    ed = P.build_ed_type(acuity, load, config)
    alpha = P.v(P.parameters()["prediction"]["alpha_main"])
    rows = run_ed_rep(ed, rep, policies_for(config), alpha=alpha, alphas=[alpha])
    for r in rows:
        r["visits_per_day"] = ed.visits_per_day
        r["spaces_total"] = ed.spaces_total
        r["spaces_second"] = ed.spaces_second
        r["physicians_by_shift"] = "/".join(map(str, ed.physicians_by_shift))
        r["second_physicians_by_shift"] = "/".join(map(str, ed.second_physicians_by_shift))
    return rows


def main():
    f = P.design()["factors"]
    reps = P.v(P.parameters()["run_control"]["replications"])
    if len(sys.argv) > 1:
        reps = int(sys.argv[1])
    tasks = [(a, l, c, r) for r in range(reps) for a in f["acuity"] for l in f["load"] for c in f["config"]]
    rows = [row for chunk in pmap(task, tasks) for row in chunk]
    df = pd.DataFrame(rows)
    save(df, "kpis", "E2_main", replications=reps, alpha=P.v(P.parameters()["prediction"]["alpha_main"]),
         ed_types=27, design="acuity x load x config; boarding low; diagnostics typical")


if __name__ == "__main__":
    main()
