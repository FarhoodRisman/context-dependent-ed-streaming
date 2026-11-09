"""E6b lever check of the optimized policies of E6 (test replications 0-9, fast KPIs).
Each optimized rule is re-run (i) as optimized, (ii) with ageing gamma set to fixed values,
because the search was centred on no ageing and rarely sampled large gamma, (iii) without the
reassessment bonus (beta = 0) and (iv) in two-area EDs without the routing offset (b = 0).
The default policy is run on the same replications for the balanced objective."""
import copy
import os

import pandas as pd

from common import P, pmap, save
from edsim.engine import Engine
from edsim.generator import generate
from edsim.kpis import fast_kpis
from edsim.policies import make_queue_rule, make_routing_rule
from opt_common import load_specs, rules_from_spec

REPS = list(range(10))
GAMMAS = [0.02, 0.05, 0.1, 0.2, 0.4]


def variants(s):
    out = [("opt", s)]
    for g in GAMMAS:
        v = copy.deepcopy(s); v["queue"]["aging"] = g; out.append((f"gamma={g}", v))
    v = copy.deepcopy(s); v["queue"]["ra_bonus"] = 0.0; out.append(("beta=0", v))
    if s["routing"]:
        v = copy.deepcopy(s); v["routing"]["offset"] = 0.0; out.append(("offset=0", v))
    return out


def task(args):
    acuity, load, config = args
    specs = [s for s in load_specs() if (s["acuity"], s["load"], s["config"]) == args]
    ed = P.build_ed_type(acuity, load, config)
    pc = P.policies_config()
    dq = make_queue_rule(pc["default_policy"]["queue"])
    dr = make_routing_rule(pc["default_policy"]["routing"]) if config != "POOLED" else None
    rows = []
    for rep in REPS:
        pts = generate(ed, rep, alphas=[1.0])
        k = fast_kpis(Engine(ed, pts, dq, dr, alpha=1.0).run())
        rows.append({"acuity": acuity, "load": load, "config": config, "rep": rep, "objective": "", "variant": "DEFAULT", **k})
        for s in specs:
            for name, v in variants(s):
                q, r = rules_from_spec(v, "L")
                k = fast_kpis(Engine(ed, pts, q, r if config != "POOLED" else None, alpha=1.0).run())
                rows.append({"acuity": acuity, "load": load, "config": config, "rep": rep,
                             "objective": s["objective"], "variant": name, **k})
    return rows


def main():
    f = P.design()["factors"]
    tasks = [(a, l, c) for c in f["config"] for a in f["acuity"] for l in f["load"]]
    if os.environ.get("EDSIM_QUICK"):
        tasks = tasks[:2]
    df = pd.DataFrame([r for rr in pmap(task, tasks) for r in rr])
    save(df, "lever_kpis", "E6b_levers", reps=REPS, gammas=GAMMAS)


if __name__ == "__main__":
    main()
