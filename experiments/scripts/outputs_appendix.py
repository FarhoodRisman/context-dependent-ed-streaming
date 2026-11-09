"""Appendix tables and figures that describe the model itself (coding scheme, confusion
matrices, worked example, inputs, arrival profile, pathways, routing-offset check).
Called by make_outputs.py; uses its macro registry and writers."""
from __future__ import annotations

import itertools
import json

import numpy as np

import make_outputs as mo
from common import P
from edsim.generator import confusion_matrices, service_cutpoints
from edsim.policies import RoutePred, RouteSimilarity, make_queue_rule, make_routing_rule, similarity

CODES = list(itertools.product(range(1, 6), range(1, 6), (1, 2), (1, 2), range(1, 5), (1, 2)))
ATTR = ["A", "B", "C", "D", "E", "F"]


def table_confusion():
    cms = confusion_matrices()
    lines = [r"\begin{tabular}{l" + "r" * 5 + "}", r"\toprule",
             r"True class & \multicolumn{5}{c}{Predicted class (service time; 1 very long \dots 5 very short)} \\",
             r"\cmidrule(lr){2-6}", r" & 1 & 2 & 3 & 4 & 5 \\", r"\midrule"]
    for i, row in enumerate(cms["service"]):
        lines.append(f"{i + 1} & " + " & ".join(f"{x:.3f}" for x in row) + r" \\")
    lines += [r"\midrule", r" & \multicolumn{5}{c}{Predicted class (resource use; 1 three or more steps \dots 4 none)} \\",
              r"\cmidrule(lr){2-5}", r" & 1 & 2 & 3 & 4 & \\", r"\midrule"]
    for i, row in enumerate(cms["resource"]):
        lines.append(f"{i + 1} & " + " & ".join(f"{x:.3f}" for x in row) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_confusion", "\n".join(lines))
    for i, c in enumerate(service_cutpoints()):
        mo.macro(f"SvcCut{['One', 'Two', 'Three', 'Four'][i]}", c, "{:.0f}")


def table_worked_example():
    """Four example patients scored by SIM, by the exact CTAS-SPT representation and by
    similarity routing (all values computed with the model's own functions)."""
    pc = P.policies_config()
    sim = pc["queue_rules"]["SIM"]
    rep = pc["similarity_representations"]["SIMREP-CTAS-SPT"]
    rt = pc["routing_rules"]["STREAM-SIM"]
    pts = [("P1", (2, 1, 1, 1, 1, 1)), ("P2", (2, 4, 2, 1, 3, 2)), ("P3", (3, 5, 2, 2, 4, 2)), ("P4", (4, 3, 2, 1, 2, 2))]
    rows = []
    for name, c in pts:
        s1 = similarity(c, sim["target"], sim["weights"])
        s2 = similarity(c, rep["target"], rep["weights"])
        d = similarity(c, rt["target_second"], rt["weights"]) - similarity(c, rt["target_main"], rt["weights"])
        rows.append((name, c, s1, s2, d))
    rank = lambda k: {r[0]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: (-r[k], r[0])))}
    r1, r2 = rank(2), rank(3)
    lines = [r"\begin{tabular}{lcrrrrrl}", r"\toprule",
             r"Patient & Code A-B-C-D-E-F & $S_{\text{SIM}}$ & Rank & $S_{\text{rep}}$ & Rank & $S_{\text{second}}-S_{\text{main}}$ & Area \\",
             r"\midrule"]
    for name, c, s1, s2, d in rows:
        area = "Main" if c[0] < 3 or d <= 0 else "Second"
        lines.append(f"{name} & {'-'.join(map(str, c))} & {s1:.4f} & {r1[name]} & {s2:.4f} & {r2[name]} & {d:+.4f} & {area} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_worked_example", "\n".join(lines))
    # macros for the text of the worked example (patient P3 under SIM)
    n, c, s1, s2, d = rows[2]
    mo.macro("WexPThreeSim", s1, "{:.4f}")
    mo.macro("WexPTwoSim", rows[1][2], "{:.4f}")
    mo.macro("WexPThreeRankSim", r1["P3"], "{:.0f}")
    mo.macro("WexPTwoRankSim", r1["P2"], "{:.0f}")


def table_inputs():
    prm = P.parameters()
    d = prm["durations"]
    rows = [("Triage", d["triage"]), ("Registration (delay)", d["registration"]),
            ("Physician reassessment", d["physician_reassessment"]), ("Laboratory (LB)", d["LB"]),
            ("X-ray (XR)", d["XR"]), ("CT", d["CT"]), ("Consultation (DR)", d["DR"]),
            ("Discharge process", d["discharge_process"]), ("Boarding, low", d["boarding"]["low"]),
            ("Boarding, high", d["boarding"]["high"])]
    lines = [r"\begin{tabular}{lrrp{6.6cm}}", r"\toprule", r"Activity & Mean (min) & CV & Source \\", r"\midrule"]
    am = d["physician_initial_by_ctas"]
    lines.append(r"Physician initial assessment & " + "/".join(str(x) for x in am["mean"]) + f" & {am['cv']:.1f} & {_src(am['source'])} \\\\")
    for nm, e in rows:
        lines.append(f"{nm} & {e['mean']:.0f} & {e['cv']:.1f} & {_src(e['source'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_inputs", "\n".join(lines))
    tilt = mo.P.v(prm["pathways"]["diagnostic_tilt_by_ctas"])
    for i, t in enumerate(tilt):
        mo.macro(f"Tilt{['One', 'Two', 'Three', 'Four', 'Five'][i]}All", t)


def _src(s: str) -> str:
    s = s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    return s if len(s) < 140 else s[:137] + "..."


def figure_arrival_profile():
    prof = P.hourly_profile()
    fig, ax = mo.plt.subplots(figsize=(3.4, 1.9))
    ax.bar(np.arange(24), prof, width=0.8, color=mo.PALETTE[0])
    for h in P.v(P.parameters()["capacity_sizing"]["shifts_start_hours"]):
        ax.axvline(h - 0.5, color=mo.MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Relative arrival rate")
    ax.set_xticks(range(0, 24, 3))
    fig.savefig(mo.FIG / "fig_arrival_profile.pdf")
    mo.plt.close(fig)
    mo.macro("ProfileMax", prof.max())
    mo.macro("ProfileMin", prof.min())


def table_pathways():
    pws = P.pathways()
    tot = sum(p.base_freq for p in pws)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & & Published & \multicolumn{3}{c}{Model probability given disposition (\%)} \\",
             r"\cmidrule(lr){4-6}", r"Pathway & Steps & share (\%) & CTAS 1 & CTAS 3 & CTAS 5 \\", r"\midrule"]
    probs = {c: {a: P.pathway_probs(c, a, "typical") for a in (True, False)} for c in (1, 3, 5)}
    for i, p in enumerate(pws):
        steps = "TR-RG-AS" + ("-" + "-".join(p.steps) if p.steps else "") + ("-AD" if p.admit else "-DD")
        lines.append(f"{p.name} & {steps} & {100 * p.base_freq / tot:.1f} & " +
                     " & ".join(f"{100 * probs[c][p.admit][i]:.1f}" for c in (1, 3, 5)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_pathways", "\n".join(lines))


def routing_offset_check():
    """Exhaustive check of the routing-offset proposition: weights on B and C only with
    w_B < 4 w_C and offset in [max(w_C, w_B - w_C), w_C + w_B/2) reproduce prediction streaming."""
    wb, wc = 0.4, 0.3
    b = 0.5 * (max(wc, wb - wc) + wc + wb / 2)
    rr = RouteSimilarity("x", [0, wb, wc, 0, 0, 0], [1] * 6, [5, 5, 2, 2, 4, 2], ctas_floor=3, offset=b)
    ref = RoutePred()
    agree = np.mean([rr.to_second(c[0], c) == ref.to_second(c[0], c) for c in CODES])
    mo.macro("PropFourAgree", 100 * agree, "{:.0f}")
    mo.macro("PropFourWB", wb); mo.macro("PropFourWC", wc); mo.macro("PropFourB", b)
    # and the default similarity routing (offset 0) for contrast
    d = make_routing_rule("STREAM-SIM")
    mo.macro("PropFourAgreeSimRoute", 100 * np.mean([d.to_second(c[0], c) == ref.to_second(c[0], c) for c in CODES]), "{:.0f}")
    mo.macro("NCodes", len(CODES), "{:,.0f}")


def build_all():
    table_confusion()
    table_worked_example()
    table_inputs()
    figure_arrival_profile()
    table_pathways()
    routing_offset_check()
