"""E4 value of prediction accuracy: prediction-error sweep on the nine medium-load core ED types.

Policies that use predicted attributes are run at every alpha; policies that do not
(CTAS and APQ with CTAS streaming, or pooled) are run once and shown as flat reference lines.
Common random numbers hold across alpha: each patient's predicted class at every alpha
is obtained from the same uniform draw.
"""
import pandas as pd

from common import P, pmap, save
from edsim.generator import generate
from edsim.runner import kpi_row, run_policy

PRED_FREE_QUEUE = {"FCFS", "CTAS", "APQ"}


def uses_prediction(q, r):
    return q not in PRED_FREE_QUEUE or (r is not None and r != "STREAM-CTAS")


def task(args):
    acuity, config, rep = args
    load = P.design()["alpha_ed_types_load"]
    ed = P.build_ed_type(acuity, load, config)
    alphas = P.v(P.parameters()["prediction"]["alpha_sweep"])
    pts = generate(ed, rep, alphas=alphas)
    qs = P.design()["alpha_queue_rules"]
    rs = [None] if config == "POOLED" else P.design()["alpha_routing_rules"]
    rows = []
    for q in qs:
        for r in rs:
            for a in (alphas if uses_prediction(q, r) else [alphas[0]]):
                eng = run_policy(ed, pts, q, r, a)
                row = kpi_row(ed, rep, q, r, a, eng)
                row["uses_prediction"] = uses_prediction(q, r)
                rows.append(row)
    return rows


def main():
    reps = P.v(P.parameters()["run_control"]["replications_secondary"])
    f = P.design()["factors"]
    tasks = [(a, c, r) for r in range(reps) for a in f["acuity"] for c in f["config"]]
    df = pd.DataFrame([row for chunk in pmap(task, tasks) for row in chunk])
    save(df, "kpis_alpha", "E4_alpha", replications=reps, load=P.design()["alpha_ed_types_load"],
         alphas=P.v(P.parameters()["prediction"]["alpha_sweep"]))


if __name__ == "__main__":
    main()
