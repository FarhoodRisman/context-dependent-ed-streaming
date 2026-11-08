"""E7 transfer of optimized policies: the policy optimized for ED type i is applied to every
other ED type j with the same configuration (test replications 0-9, fast KPIs). The default
policy is run on the same replications so that both objectives can be computed."""
import os

import pandas as pd

from common import P, pmap, save
from edsim.engine import Engine
from edsim.generator import generate
from edsim.kpis import fast_kpis
from edsim.policies import make_queue_rule, make_routing_rule
from opt_common import load_specs, rules_from_spec

REPS = list(range(10))


def task(args):
    acuity, load, config = args          # the ED type j where policies are applied
    specs = [s for s in load_specs() if s["config"] == config]
    ed = P.build_ed_type(acuity, load, config)
    pc = P.policies_config()
    dq = make_queue_rule(pc["default_policy"]["queue"])
    dr = make_routing_rule(pc["default_policy"]["routing"]) if config != "POOLED" else None
    rows = []
    for rep in REPS:
        pts = generate(ed, rep, alphas=[1.0])
        k = fast_kpis(Engine(ed, pts, dq, dr, alpha=1.0).run())
        rows.append({"acuity": acuity, "load": load, "config": config, "rep": rep, "source": "DEFAULT", "objective": "", **k})
        for s in specs:
            q, r = rules_from_spec(s, "T")
            k = fast_kpis(Engine(ed, pts, q, r if config != "POOLED" else None, alpha=1.0).run())
            rows.append({"acuity": acuity, "load": load, "config": config, "rep": rep,
                         "source": f"{s['acuity']}|{s['load']}", "objective": s["objective"], **k})
    return rows


def main():
    f = P.design()["factors"]
    tasks = [(a, l, c) for c in f["config"] for a in f["acuity"] for l in f["load"]]
    if os.environ.get("EDSIM_QUICK"):     # pilot runs only
        tasks = tasks[:2]
    df = pd.DataFrame([r for rr in pmap(task, tasks) for r in rr])
    save(df, "transfer_kpis", "E7_transfer", reps=REPS)


if __name__ == "__main__":
    main()
