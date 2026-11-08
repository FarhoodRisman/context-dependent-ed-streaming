"""E1 verification and validation.

(a) Process logic: stage-to-stage transition frequencies recovered from the event log
    are compared with (i) the transitions implied by the generator's pathway
    probabilities for this ED type (verification) and (ii) the transitions implied by
    the published pathway table (Rismanchian2023_PatientJourney), renormalised
    over the 11 pathways (validation of the pathway mix).
(b) Face validity: door-to-physician and LOS percentiles of the community ED under
    the default policy at each load level, to be set against published Canadian
    aggregates in the paper.
"""
from collections import Counter

import numpy as np
import pandas as pd

from common import P, pmap, save
from edsim.generator import generate
from edsim.kpis import event_log_frame
from edsim.runner import kpi_row, run_policy

DEFAULT = P.policies_config()["default_policy"]


def sequences_from_log(log, pids):
    g = log[log.pid.isin(pids) & log.stage.isin(["TRIAGE", "REG", "AS", "LB", "XR", "CT", "DR", "DD", "BOARD"])]
    g = g.sort_values(["pid", "t_start", "t_end"])
    names = {"TRIAGE": "TR", "REG": "RG", "BOARD": "AD"}
    seqs = g.groupby("pid").stage.apply(lambda s: [names.get(x, x) for x in s])
    return seqs


def transitions(seqs_with_weights):
    c = Counter()
    for seq, w in seqs_with_weights:
        for a, b in zip(seq[:-1], seq[1:]):
            c[(a, b)] += w
    tot_from = Counter()
    for (a, _), w in c.items():
        tot_from[a] += w
    return {k: w / tot_from[k[0]] for k, w in c.items()}


def task(args):
    load, rep = args
    ed = P.build_ed_type("community", load, "POOLED")
    pts = generate(ed, rep, alphas=[1.0])
    eng = run_policy(ed, pts, DEFAULT["queue"], None)
    row = kpi_row(ed, rep, DEFAULT["queue"], None, 1.0, eng)
    out = {"kpi": row}
    if load == "medium":
        log = event_log_frame(eng)
        obs = [p.pid for p in pts if eng.warm <= p.arrival < eng.obs_end]
        seqs = sequences_from_log(log, obs)
        out["seqs"] = Counter(tuple(s) for s in seqs)
        out["pathways"] = Counter(pts[i].pathway for i in obs)
    return out


def main():
    reps = P.v(P.parameters()["run_control"]["replications"])
    tasks = [(l, r) for l in P.design()["factors"]["load"] for r in range(reps)]
    res = pmap(task, tasks)
    kp = pd.DataFrame([r["kpi"] for r in res])
    save(kp, "face_validity_kpis", "E1_validation", replications=reps, ed_type="community|*|POOLED", policy=DEFAULT)

    # transitions: simulated
    sim = Counter()
    for r in res:
        if "seqs" in r:
            sim.update(r["seqs"])
    t_sim = transitions(sim.items())
    # expected from generator probabilities (community, typical)
    mix = P.v(P.parameters()["acuity_mix"]["community"])
    padm = P.v(P.parameters()["admission_probability_by_ctas"])
    pws = P.pathways()
    exp_w = Counter()
    for c in range(1, 6):
        for adm, pa in ((True, padm[c - 1]), (False, 1 - padm[c - 1])):
            for pw, pp in zip(pws, P.pathway_probs(c, adm, "typical")):
                if pp > 0:
                    seq = ("TR", "RG", "AS") + pw.steps + (("AD",) if pw.admit else ("DD",))
                    exp_w[seq] += mix[c - 1] * pa * pp
    t_exp = transitions(exp_w.items())
    # published table
    pub_w = Counter()
    for pw in pws:
        seq = ("TR", "RG", "AS") + pw.steps + (("AD",) if pw.admit else ("DD",))
        pub_w[seq] += pw.base_freq
    t_pub = transitions(pub_w.items())
    keys = sorted(set(t_sim) | set(t_exp) | set(t_pub))
    tr = pd.DataFrame([{"from": a, "to": b, "simulated": t_sim.get((a, b), 0.0),
                        "expected_generator": t_exp.get((a, b), 0.0), "published_table": t_pub.get((a, b), 0.0)}
                       for a, b in keys])
    save(tr, "transitions", "E1_validation", replications=reps, ed_type="community|medium|POOLED")

    # pathway shares
    pw_sim = Counter()
    for r in res:
        if "pathways" in r:
            pw_sim.update(r["pathways"])
    n = sum(pw_sim.values())
    tot_pub = sum(pw.base_freq for pw in pws)
    exp_share = Counter()
    for seq, w in exp_w.items():
        exp_share[seq] += w
    rows = []
    for pw in pws:
        seq = ("TR", "RG", "AS") + pw.steps + (("AD",) if pw.admit else ("DD",))
        rows.append({"pathway": pw.name, "sequence": ">".join(seq), "simulated_share": pw_sim[pw.name] / n,
                     "expected_generator": exp_share[seq], "published_share_renormalised": pw.base_freq / tot_pub})
    save(pd.DataFrame(rows), "pathway_shares", "E1_validation", replications=reps, ed_type="community|medium|POOLED")


if __name__ == "__main__":
    main()
