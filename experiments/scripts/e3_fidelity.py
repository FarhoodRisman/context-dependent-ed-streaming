"""E3 policy fidelity: does a similarity rule apply the policy it is meant to represent?

Decision level: the engine runs the similarity representation and, at every dispatch
from a queue with two or more waiting patients, records whether the coded reference
rule would have chosen the same patient; Kendall's tau between the two queue orders
is computed at every fifth such dispatch with three or more patients. Routing
fidelity is recorded at every routing decision.
Outcome level: the reference rule is also run on the same patient list and the
KPIs are compared pairwise.
"""
import pandas as pd

from common import P, pmap, save
from edsim.engine import Engine
from edsim.generator import generate
from edsim.kpis import compute_kpis
from edsim.policies import make_queue_rule, make_routing_rule

QUEUE_PAIRS = [("SIMREP-CTAS", "CTAS"), ("SIMREP-CTAS-SPT", "CTAS-SPT"), ("SIMREP-CTAS-PD", "CTAS-PD"),
               ("SIMREP-NEAR", "CTAS-SPT"), ("SIM", "CTAS-SPT")]
ROUTE_PAIRS = [("ROUTE-SIMREP-CTAS", "STREAM-CTAS"), ("STREAM-SIM", "STREAM-PRED")]
KPIS = ["los_all_mean", "pia_all_mean", "pia_all_p90", "ratp", "awt", "pia_ctas2_p95", "los_disch_mean"]


def task(args):
    acuity, load, config, rep = args
    ed = P.build_ed_type(acuity, load, config)
    alpha = P.v(P.parameters()["prediction"]["alpha_main"])
    pts = generate(ed, rep, alphas=[alpha])
    rows = []
    for rep_name, ref_name in QUEUE_PAIRS:
        rr = make_routing_rule("STREAM-CTAS")
        e_rep = Engine(ed, pts, make_queue_rule(rep_name), rr, alpha=alpha,
                       shadow=make_queue_rule(ref_name)).run()
        e_ref = Engine(ed, pts, make_queue_rule(ref_name), make_routing_rule("STREAM-CTAS"), alpha=alpha).run()
        k_rep, k_ref = compute_kpis(e_rep), compute_kpis(e_ref)
        tau = e_rep.fid["tau"]
        row = {"acuity": acuity, "load": load, "config": config, "rep": rep, "level": "queue",
               "representation": rep_name, "reference": ref_name,
               "decisions": e_rep.fid["n"], "agree": e_rep.fid["agree"],
               "tau_mean": sum(tau) / len(tau) if tau else float("nan"), "tau_n": len(tau),
               "identical_log": e_rep.log == e_ref.log}
        for k in KPIS:
            row[f"{k}_rep"], row[f"{k}_ref"] = k_rep[k], k_ref[k]
        rows.append(row)
    for rep_name, ref_name in ROUTE_PAIRS:
        e_rep = Engine(ed, pts, make_queue_rule("CTAS"), make_routing_rule(rep_name), alpha=alpha,
                       shadow_routing=make_routing_rule(ref_name)).run()
        e_ref = Engine(ed, pts, make_queue_rule("CTAS"), make_routing_rule(ref_name), alpha=alpha).run()
        k_rep, k_ref = compute_kpis(e_rep), compute_kpis(e_ref)
        row = {"acuity": acuity, "load": load, "config": config, "rep": rep, "level": "routing",
               "representation": rep_name, "reference": ref_name,
               "decisions": e_rep.fid_route["n"], "agree": e_rep.fid_route["agree"],
               "tau_mean": float("nan"), "tau_n": 0, "identical_log": e_rep.log == e_ref.log}
        for k in KPIS:
            row[f"{k}_rep"], row[f"{k}_ref"] = k_rep[k], k_ref[k]
        rows.append(row)
    return rows


def main():
    reps = P.v(P.parameters()["run_control"]["replications_secondary"])
    types = P.design()["fidelity_ed_types"]
    tasks = [(a, l, c, r) for r in range(reps) for a, l, c in types]
    df = pd.DataFrame([row for chunk in pmap(task, tasks) for row in chunk])
    save(df, "fidelity", "E3_fidelity", replications=reps, ed_types=types,
         queue_pairs=QUEUE_PAIRS, routing_pairs=ROUTE_PAIRS)


if __name__ == "__main__":
    main()
