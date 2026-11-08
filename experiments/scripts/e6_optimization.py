"""E6 simulation-based optimization of similarity policies (cross-entropy method).

For each of the 27 core ED types and two objectives ('access': RATP; 'balanced': mean of
RATP and mean LOS, each relative to the default policy on the same replications), the CEM
searches queue weights, target codes, ageing and reassessment bonus, and in two-area EDs the
routing weights and offset. Replications: training (search) 1000-1002, validation (selection)
2000-2004, test 0-24 (the E2 replications, so the optimized policy is compared pairwise with
every fixed policy on identical patient lists).
"""
import json
import os
import sys

import pandas as pd

from common import P, outdir, pmap, save
from edsim.engine import Engine
from edsim.generator import generate
from edsim.optimize import CEMOptimizer, CEMSettings, build_rules
from edsim.policies import make_queue_rule, make_routing_rule
from edsim.runner import kpi_row

TRAIN = [1000, 1001, 1002]
VAL = [2000, 2001, 2002, 2003, 2004]
OBJECTIVES = ["access", "balanced"]
SETTINGS = CEMSettings()
if os.environ.get("EDSIM_QUICK"):        # pilot runs only; never used for reported results
    SETTINGS = CEMSettings(iterations=2, population=4, elite=2, shortlist=2)


def task(args):
    acuity, load, config, objective = args
    ed = P.build_ed_type(acuity, load, config)
    pc = P.policies_config()
    init_q = pc["queue_rules"]["SIM"]
    init_r = pc["routing_rules"]["STREAM-SIM"] if config != "POOLED" else None
    dflt = (make_queue_rule(pc["default_policy"]["queue"]),
            make_routing_rule(pc["default_policy"]["routing"]) if config != "POOLED" else None)
    train = [generate(ed, r, alphas=[1.0]) for r in TRAIN]
    val = [generate(ed, r, alphas=[1.0]) for r in VAL]
    f = P.design()["factors"]
    seed = 7000 + 100 * f["acuity"].index(acuity) + 10 * f["load"].index(load) + f["config"].index(config) \
        + 1000 * OBJECTIVES.index(objective)
    opt = CEMOptimizer(ed, objective, train, val, SETTINGS, seed, init_q, init_r, dflt)
    best, info = opt.search()
    qs, rs = best.queue_spec(), best.routing_spec()
    name = "OPT-" + objective.upper()
    rows = []
    reps = P.v(P.parameters()["run_control"]["replications"])
    if os.environ.get("EDSIM_QUICK"):
        reps = 2
    for rep in range(reps):
        pts = generate(ed, rep, alphas=[1.0])
        q, r = build_rules(qs, rs, name)
        eng = Engine(ed, pts, q, r if config != "POOLED" else None, alpha=1.0).run()
        row = kpi_row(ed, rep, name, name + "-R" if config != "POOLED" else None, 1.0, eng)
        row["objective"] = objective
        rows.append(row)
    spec = {"acuity": acuity, "load": load, "config": config, "objective": objective,
            "queue": qs, "routing": rs, **info, "history": opt.state.history}
    return rows, spec


def main():
    f = P.design()["factors"]
    tasks = [(a, l, c, o) for o in OBJECTIVES for a in f["acuity"] for l in f["load"] for c in f["config"]]
    if len(sys.argv) > 1:          # quick check: limit the number of tasks
        tasks = tasks[: int(sys.argv[1])]
    res = pmap(task, tasks)
    rows = [r for rr, _ in res for r in rr]
    specs = [s for _, s in res]
    save(pd.DataFrame(rows), "opt_test_kpis", "E6_optimization", objectives=OBJECTIVES, train_reps=TRAIN,
         validation_reps=VAL, cem=SETTINGS.__dict__)
    d = outdir("E6_optimization")
    (d / "opt_specs.json").write_text(json.dumps(specs, indent=1))
    hist = pd.DataFrame([{"acuity": s["acuity"], "load": s["load"], "config": s["config"],
                          "objective": s["objective"], **h} for s in specs for h in s["history"]])
    save(hist, "opt_history", "E6_optimization", cem=SETTINGS.__dict__)


if __name__ == "__main__":
    main()
