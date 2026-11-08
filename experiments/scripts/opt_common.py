"""Helpers shared by the experiments that reuse the optimized policies of E6."""
import json

from common import P, RESULTS
from edsim.optimize import build_rules


def load_specs():
    p = RESULTS / "E6_optimization" / "opt_specs.json"
    return json.loads(p.read_text()) if p.exists() else []


def spec_for(specs, acuity, load, config, objective):
    for s in specs:
        if (s["acuity"], s["load"], s["config"], s["objective"]) == (acuity, load, config, objective):
            return s
    return None


def rules_from_spec(s, name):
    return build_rules(s["queue"], s["routing"], name)


def objective_from_kpis(objective, k, ref):
    if objective == "access":
        return k["ratp"]
    return 0.5 * k["ratp"] / ref["ratp"] + 0.5 * k["los_all_mean"] / ref["los_all_mean"]
