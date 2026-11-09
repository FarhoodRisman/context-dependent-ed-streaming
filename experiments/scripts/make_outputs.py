"""Build every table (.tex), figure (.pdf) and paper/numbers.tex from the raw results.

Nothing in the manuscript's results is typed by hand: the text uses the macros in
numbers.tex, and tables/figures are \\input / \\includegraphics of the files written here.
Run check_results.py first (this script calls it and stops if a check fails).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from common import P, RESULTS, ROOT, provenance  # noqa: E402

PAPER = ROOT / "paper"
TAB = PAPER / "tables"
FIG = PAPER / "figures"
SUPP = PAPER / "supplement"
for d in (TAB, FIG, SUPP):
    d.mkdir(parents=True, exist_ok=True)

TYPE = ["acuity", "load", "config"]
ACU = ["tertiary", "community", "urgentcare"]
ACU_LAB = {"tertiary": "Tertiary", "community": "Community", "urgentcare": "Urgent-care-like"}
LOAD = ["low", "medium", "high"]
CFG = ["POOLED", "FASTTRACK", "SPLITFLOW"]
CFG_LAB = {"POOLED": "Pooled", "FASTTRACK": "Fast track", "SPLITFLOW": "Split flow"}
QUEUES = list(P.policies_config()["queue_rules"].keys())
ROUTES = list(P.policies_config()["routing_rules"].keys())
RLAB = {"STREAM-CTAS": "C", "STREAM-PRED": "P", "STREAM-SIM": "S", "": ""}
# categorical palette (validated with the dataviz validator, light mode): fixed order by queue rule
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
QCOL = dict(zip(QUEUES, PALETTE))
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e3"
KPIS = {"ratp": "RATP", "los_all_mean": "Mean LOS", "awt": "AWT", "pia_all_p90": "P90 door to physician",
        "pia_ctas2_p95": "P95 door to physician, CTAS 2", "pia_ctas5_p95": "P95 door to physician, CTAS 5"}
DEFAULT = P.policies_config()["default_policy"]
T975 = lambda n: stats.t.ppf(0.975, n - 1)
OVERLOAD = 0.98   # mean Main-space utilisation at or above this: queues grow through the run (flagged, not interpreted)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})

MACROS: dict[str, str] = {}


# ------------------------------------------------------------------ helpers
def macro(name: str, value, fmt="{:.2f}"):
    assert name.isalpha(), name
    assert name not in MACROS, name
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def cname(s: str) -> str:
    """CamelCase letters-only name for macro keys."""
    rep = {"-": " ", "|": " ", "_": " ", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
           "9": "Nine", "0": "Zero", ".": " "}
    for a, b in rep.items():
        s = s.replace(a, b)
    return "".join(w[:1].upper() + w[1:].lower() for w in s.split())


def plabel(q, r):
    return q if not r else f"{q}/{RLAB[r]}"


def tlabel(a, l, c):
    return f"{ACU_LAB[a]}, {l}, {CFG_LAB[c]}"


def pct(x):
    return 100.0 * x


def write_tex(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")


def fmt_ci(m, h, f="{:.2f}"):
    return (f + r" $\pm$ " + f).format(m, h)


# ------------------------------------------------------------------ data
def load():
    e2 = pd.read_csv(RESULTS / "E2_main/kpis.csv.gz")
    e2["routing"] = e2["routing"].fillna("")
    return e2


def policy_table(e2):
    """Mean of each KPI by ED type and policy, plus replication count."""
    g = e2.groupby(TYPE + ["queue", "routing"])
    cols = list(dict.fromkeys(list(KPIS) + ["los_disch_mean", "los_adm_mean", "pia_ctas2_p95", "pia_ctas3_p95",
                                            "pia_ctas4_p95", "pia_ctas5_p95", "pia_all_mean", "visits_per_day",
                                            "util_MAIN", "share_second_area", "pia_ctas3_mean", "pia_ctas4_mean"]))
    m = g[cols].mean().reset_index()
    m["overloaded"] = m.util_MAIN >= OVERLOAD
    m["n"] = g.size().values
    return m


def paired(e2, typ, k, a, b):
    """Mean and 95% half-width of K(a) - K(b) across replications for one ED type (paired by rep)."""
    d = e2[(e2.acuity == typ[0]) & (e2.load == typ[1]) & (e2.config == typ[2])]
    x = d[(d.queue == a[0]) & (d.routing == a[1])].set_index("rep")[k]
    y = d[(d.queue == b[0]) & (d.routing == b[1])].set_index("rep")[k]
    z = (x - y).dropna()
    n = len(z)
    return float(z.mean()), float(T975(n) * z.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), n


def default_for(config):
    return (DEFAULT["queue"], "" if config == "POOLED" else DEFAULT["routing"])


def candidates_for(config):
    if config == "POOLED":
        return [(q, "") for q in QUEUES]
    return [(q, r) for q in QUEUES for r in ROUTES]


# ------------------------------------------------------------------ analyses
def best_by_type(e2, m):
    rows = []
    for (a, l, c), g in m.groupby(TYPE):
        for k in KPIS:
            i = g[k].idxmin()
            bq, br = g.loc[i, "queue"], g.loc[i, "routing"]
            tied = 0
            for q, r in candidates_for(c):
                if (q, r) == (bq, br):
                    tied += 1
                    continue
                mu, hw, _ = paired(e2, (a, l, c), k, (q, r), (bq, br))
                if mu - hw <= 0:
                    tied += 1
            dq, dr = default_for(c)
            dval = float(g[(g.queue == dq) & (g.routing == dr)][k].iloc[0])
            mu, hw, n = paired(e2, (a, l, c), k, (bq, br), (dq, dr))
            rows.append({"acuity": a, "load": l, "config": c, "kpi": k, "best_queue": bq, "best_routing": br,
                         "best": float(g.loc[i, k]), "default": dval, "best_minus_default": mu,
                         "best_minus_default_hw": hw, "n_tied": tied, "n_policies": len(candidates_for(c)), "reps": n})
    return pd.DataFrame(rows)


def regret(m):
    """Relative regret of every single (queue, routing) policy used in all 27 ED types."""
    rows = []
    for k in KPIS:
        best = m.groupby(TYPE)[k].min()
        for q in QUEUES:
            for r in ROUTES:
                for (a, l, c), b in best.items():
                    rr = "" if c == "POOLED" else r
                    v = float(m[(m.acuity == a) & (m.load == l) & (m.config == c) & (m.queue == q) & (m.routing == rr)][k].iloc[0])
                    rows.append({"kpi": k, "queue": q, "routing": r, "acuity": a, "load": l, "config": c,
                                 "value": v, "best": float(b), "regret_abs": v - b, "regret_rel": v / b - 1 if b > 0 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tables
def table_parameters():
    prm = P.parameters()
    mix = prm["acuity_mix"]
    adm = P.v(prm["admission_probability_by_ctas"])
    ub = P.v(prm["ctas_targets_minutes"])
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r" & CTAS 1 & CTAS 2 & CTAS 3 & CTAS 4 & CTAS 5 \\", r"\midrule"]
    for a in ACU:
        vals = " & ".join(f"{pct(x):.1f}" for x in P.v(mix[a]))
        lines.append(f"{ACU_LAB[a]} mix (\\%) & {vals} \\\\")
    lines.append(r"Admission probability (\%) & " + " & ".join(f"{pct(x):.1f}" for x in adm) + r" \\")
    lines.append(r"Target, door to physician (min) & " + " & ".join(str(x) for x in ub) + r" \\")
    am = prm["durations"]["physician_initial_by_ctas"]["mean"]
    lines.append(r"Mean initial assessment (min) & " + " & ".join(str(x) for x in am) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_parameters", "\n".join(lines))
    d = prm["durations"]
    for key, nm in (("triage", "Triage"), ("registration", "Registration"), ("physician_reassessment", "Reassess"),
                    ("LB", "Lab"), ("XR", "Xray"), ("CT", "CT"), ("DR", "Consult"), ("discharge_process", "Discharge")):
        macro(f"Dur{nm}Mean", d[key]["mean"], "{:.0f}")
        macro(f"Dur{nm}CV", d[key]["cv"], "{:.1f}")
    macro("DurPhysCV", d["physician_initial_by_ctas"]["cv"], "{:.1f}")
    macro("BoardLowMean", d["boarding"]["low"]["mean"], "{:.0f}")
    macro("BoardHighMean", d["boarding"]["high"]["mean"], "{:.0f}")
    macro("BoardCV", d["boarding"]["low"]["cv"], "{:.1f}")
    macro("BaseVisits", P.v(prm["arrivals"]["base_visits_per_day"]), "{:.0f}")
    cs = prm["capacity_sizing"]
    macro("PhysTargetMedium", P.v(cs["physician_target_utilisation_medium"]))
    macro("SpaceSizing", P.v(cs["space_sizing_utilisation"]))
    macro("DiagTarget", P.v(cs["diagnostic_target_utilisation"]))
    macro("TriageTarget", P.v(cs["triage_target_utilisation_peak"]))
    for l, x in P.v(prm["load_levels"]).items():
        macro(f"Load{cname(l)}", x)
    rc = prm["run_control"]
    macro("WarmupDays", P.v(rc["warmup_days"]), "{:.0f}")
    macro("ObsDays", P.v(rc["observation_days"]), "{:.0f}")
    macro("CooldownDays", P.v(rc["max_cooldown_days"]), "{:.0f}")
    macro("Reps", P.v(rc["replications"]), "{:.0f}")
    macro("RepsSecondary", P.v(rc["replications_secondary"]), "{:.0f}")
    pr = prm["prediction"]
    for k, nm in (("admission", "Adm"), ("service", "Svc"), ("diagnostics", "Diag"), ("consult", "Cons"), ("resource", "Res")):
        macro(f"Acc{nm}", pct(P.v(pr["reference_accuracy"][k])), "{:.0f}")
    macro("ErrDecay", P.v(pr["ordinal_error_decay"]))
    al = P.v(pr["alpha_sweep"])
    macro("AlphaMax", max(al))
    for k, nm in (("admission", "Adm"), ("service", "Svc"), ("diagnostics", "Diag"), ("consult", "Cons")):
        macro(f"Acc{nm}AtMax", pct(1 - max(al) * (1 - P.v(pr["reference_accuracy"][k]))), "{:.1f}")
    macro("AlphaStep", al[1] - al[0])
    tilt = P.v(prm["pathways"]["diagnostic_tilt_by_ctas"])
    macro("TiltOne", tilt[0]); macro("TiltFive", tilt[4])
    sh = P.v(prm["pathways"]["diagnostic_intensity_shift"])
    macro("DiagShiftHeavy", sh["heavy"])
    q = P.policies_config()["queue_rules"]
    w = q["SIM"]["weights"]
    for nm, x in zip(["A", "B", "C", "D", "E", "F"], w):
        macro(f"SimW{nm}", x)
    macro("SimAging", q["SIM-AGE"]["aging"])
    sr = P.policies_config()["similarity_representations"]
    macro("RepSptWA", sr["SIMREP-CTAS-SPT"]["weights"][0]); macro("RepSptWB", sr["SIMREP-CTAS-SPT"]["weights"][1])
    macro("NearW", sr["SIMREP-NEAR"]["weights"][0])
    # pathway table facts
    pws = P.pathways()
    tot = sum(p.base_freq for p in pws)
    macro("NPathways", len(pws), "{:.0f}")
    vt = json.loads((P.CONFIG_DIR / "validation_targets.json").read_text())
    macro("CihiNonAdmPNinety", vt["cihi_nonadmitted_less_urgent_p90_hours"]["value"], "{:.1f}")
    macro("CihiAdmPNinety", vt["cihi_admitted_p90_hours"]["value"], "{:.1f}")
    bm = vt["provincial_median_wait_inpatient_bed_hours"]["value"]
    macro("ProvBedWaitLow", bm[0], "{:.1f}"); macro("ProvBedWaitHigh", bm[1], "{:.1f}")


def table_capacity():
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Visits per day} & Spaces & Physicians \\",
             r"\cmidrule(lr){3-5}",
             r"Acuity mix & Configuration & Low & Medium & High & (second area) & per shift (second area) \\", r"\midrule"]
    for a in ACU:
        for c in CFG:
            eds = [P.build_ed_type(a, l, c) for l in LOAD]
            e = eds[1]
            sp = f"{e.spaces_total}" + (f" ({e.spaces_second})" if c != "POOLED" else "")
            ph = "/".join(map(str, e.physicians_by_shift))
            if c == "FASTTRACK":
                ph += " (" + "/".join(map(str, e.second_physicians_by_shift)) + ")"
            lines.append(f"{ACU_LAB[a] if c == 'POOLED' else ''} & {CFG_LAB[c]} & " +
                         " & ".join(f"{x.visits_per_day:.0f}" for x in eds) + f" & {sp} & {ph} \\\\")
        if a != ACU[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_capacity", "\n".join(lines))
    vis = [P.build_ed_type(a, l, c).visits_per_day for a in ACU for l in LOAD for c in CFG]
    macro("VisitsMin", min(vis), "{:.0f}"); macro("VisitsMax", max(vis), "{:.0f}")


def table_validation():
    tr = pd.read_csv(RESULTS / "E1_validation/transitions.csv.gz")
    tr = tr[~tr["from"].isin(["TR", "RG"])]
    macro("TransMaxDiffGen", float((tr.simulated - tr.expected_generator).abs().max()), "{:.3f}")
    macro("TransMaxDiffPub", float((tr.simulated - tr.published_table).abs().max()), "{:.3f}")
    pw = pd.read_csv(RESULTS / "E1_validation/pathway_shares.csv.gz")
    macro("PathMaxDiffGen", float((pw.simulated_share - pw.expected_generator).abs().max()), "{:.3f}")
    macro("PathMaxDiffPub", float((pw.simulated_share - pw.published_share_renormalised).abs().max()), "{:.3f}")
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"From & To & Simulated & Generator & Published \\", r"\midrule"]
    for _, r in tr.iterrows():
        lines.append(f"{r['from']} & {r['to']} & {r.simulated:.3f} & {r.expected_generator:.3f} & {r.published_table:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_transitions", "\n".join(lines))
    fv = pd.read_csv(RESULTS / "E1_validation/face_validity_kpis.csv.gz")
    g = fv.groupby("load").mean(numeric_only=True)
    cols = [("pia_all_p50", "Door to physician, median"), ("pia_all_p90", "Door to physician, P90"),
            ("los_disch_p50", "LOS discharged, median"), ("los_disch_p90", "LOS discharged, P90"),
            ("los_ctas4_p90", "LOS CTAS 4, P90"), ("los_adm_p50", "LOS admitted, median"),
            ("los_adm_p90", "LOS admitted, P90"), ("util_PHYS-MAIN", "Physician utilisation"),
            ("util_MAIN", "Space utilisation")]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Measure (h unless stated) & Low & Medium & High \\", r"\midrule"]
    for k, lab in cols:
        f = (lambda x: f"{x:.2f}") if k.startswith("util") else (lambda x: f"{x / 60:.1f}")
        lines.append(f"{lab} & " + " & ".join(f(g.loc[l, k]) for l in LOAD) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_face_validity", "\n".join(lines))
    for l in LOAD:
        macro(f"FvPiaMed{cname(l)}", g.loc[l, "pia_all_p50"] / 60, "{:.1f}")
        macro(f"FvPiaPNinety{cname(l)}", g.loc[l, "pia_all_p90"] / 60, "{:.1f}")
        macro(f"FvLosCtasFourPNinety{cname(l)}", g.loc[l, "los_ctas4_p90"] / 60, "{:.1f}")
        macro(f"FvLosAdmPNinety{cname(l)}", g.loc[l, "los_adm_p90"] / 60, "{:.1f}")
    w = pd.read_csv(RESULTS / "E0_pilot/warmup_mser5.csv.gz")
    macro("MserMaxHours", w.mser5_truncation_hours.max(), "{:.0f}")
    rn = pd.read_csv(RESULTS / "E0_pilot/replications_needed.csv.gz")
    macro("RepsNeededLosMax", rn.los_all_mean_n_needed.max(), "{:.0f}")


def table_fidelity():
    f = pd.read_csv(RESULTS / "E3_fidelity/fidelity.csv.gz")
    f["agree_rate"] = f.agree / f.decisions
    order = [("queue", "SIMREP-CTAS", "CTAS"), ("queue", "SIMREP-CTAS-SPT", "CTAS-SPT"), ("queue", "SIMREP-CTAS-PD", "CTAS-PD"),
             ("routing", "ROUTE-SIMREP-CTAS", "STREAM-CTAS"), ("queue", "SIMREP-NEAR", "CTAS-SPT"), ("queue", "SIM", "CTAS-SPT"),
             ("routing", "STREAM-SIM", "STREAM-PRED")]
    lab = {"SIMREP-CTAS": r"$w=(1,0,\dots)$", "SIMREP-CTAS-SPT": rf"$w_A={MACROS['RepSptWA']}, w_B={MACROS['RepSptWB']}$",
           "SIMREP-CTAS-PD": rf"$w_A={MACROS['RepSptWA']}, w_C={MACROS['RepSptWB']}$", "ROUTE-SIMREP-CTAS": r"routing, $w=(1,0,\dots)$",
           "SIMREP-NEAR": rf"$w_A=w_B={MACROS['NearW']}$", "SIM": "SIM weights", "STREAM-SIM": "similarity routing"}
    rlab = {"CTAS": "CTAS", "CTAS-SPT": "CTAS-SPT", "CTAS-PD": "CTAS-PD", "STREAM-CTAS": "CTAS streaming",
            "STREAM-PRED": "prediction streaming"}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Similarity setting & Reference rule & Prop. & Agreement (\%) & Kendall $\tau$ & Identical logs (\%) & $\Delta$ mean LOS (\%) \\",
             r"\midrule"]
    for lvl, rep, ref in order:
        d = f[(f.level == lvl) & (f.representation == rep)]
        ag = d.agree_rate.mean()
        tau = d.tau_mean.mean()
        ident = d.identical_log.mean()
        dl = ((d.los_all_mean_rep - d.los_all_mean_ref) / d.los_all_mean_ref)
        n = len(dl)
        hw = T975(n) * dl.std(ddof=1) / math.sqrt(n)
        exact = rep.startswith("SIMREP-CTAS") or rep == "ROUTE-SIMREP-CTAS"
        prop = "2" if lvl == "queue" else "3"
        lines.append(f"{lab[rep]} & {rlab[ref]} & {prop}{' (holds)' if exact else ' (fails)'} & {pct(ag):.1f} & "
                     + ("--" if lvl == "routing" else f"{tau:.2f}") + f" & {pct(ident):.0f} & "
                     + (r"$0$" if exact and ident == 1 else f"{pct(dl.mean()):+.1f} $\\pm$ {pct(hw):.1f}") + r" \\")
        key = cname(rep)
        macro(f"Fid{key}Agree", pct(ag), "{:.1f}")
        macro(f"Fid{key}AgreeMin", pct(d.groupby("acuity").agree_rate.mean().min()), "{:.0f}")
        macro(f"Fid{key}AgreeMax", pct(d.groupby("acuity").agree_rate.mean().max()), "{:.0f}")
        if lvl == "queue":
            macro(f"Fid{key}Tau", tau)
        macro(f"Fid{key}DLos", pct(dl.mean()), "{:.1f}")
        macro(f"Fid{key}DLosAbs", abs(pct(dl.mean())), "{:.1f}")
        dr = (d.ratp_rep - d.ratp_ref)
        macro(f"Fid{key}DRatp", dr.mean(), "{:.3f}")
        macro(f"Fid{key}DRatpAbs", abs(dr.mean()), "{:.3f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_fidelity", "\n".join(lines))
    macro("FidDecisionsPerRun", f[f.level == "queue"].decisions.mean(), "{:,.0f}")
    macro("FidReps", f.rep.nunique(), "{:.0f}")
    # LOS of prediction streaming vs similarity routing in the urgent-care fast track
    d = f[(f.representation == "STREAM-SIM") & (f.acuity == "urgentcare")]
    macro("FidPredLosUrgent", d.los_all_mean_ref.mean() / 60, "{:.0f}")
    macro("FidSimLosUrgent", d.los_all_mean_rep.mean() / 60, "{:.1f}")


def table_best(best):
    lines = [r"\begin{tabular}{lllrlrr lrr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{RATP} & \multicolumn{3}{c}{Mean LOS (min)} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
             r"Acuity & Load & Config. & Default & Best policy & Best & Tied & Best policy & Best & Default \\", r"\midrule"]
    for a in ACU:
        for l in LOAD:
            for c in CFG:
                r1 = best[(best.acuity == a) & (best.load == l) & (best.config == c) & (best.kpi == "ratp")].iloc[0]
                r2 = best[(best.acuity == a) & (best.load == l) & (best.config == c) & (best.kpi == "los_all_mean")].iloc[0]
                lines.append(f"{ACU_LAB[a] if (l == 'low' and c == 'POOLED') else ''} & {l if c == 'POOLED' else ''} & {CFG_LAB[c]} & "
                             f"{r1.default:.2f} & {plabel(r1.best_queue, r1.best_routing)} & {r1.best:.2f} & {r1.n_tied}/{r1.n_policies} & "
                             f"{plabel(r2.best_queue, r2.best_routing)} & {r2.best:.0f} & {r2.default:.0f} \\\\")
            if not (a == ACU[-1] and l == LOAD[-1]):
                lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_best", "\n".join(lines))


def figure_heatmap(best):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    for ax, k, title in zip(axes, ["ratp", "los_all_mean"], ["(a) Lowest RATP", "(b) Lowest mean LOS"]):
        ax.grid(False)
        rows = [(a, l) for a in ACU for l in LOAD]
        for i, (a, l) in enumerate(rows):
            for j, c in enumerate(CFG):
                r = best[(best.acuity == a) & (best.load == l) & (best.config == c) & (best.kpi == k)].iloc[0]
                col = QCOL[r.best_queue]
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=col, ec="white", lw=1.5))
                txt = plabel(r.best_queue, r.best_routing) + f"\n{r.n_tied} of {r.n_policies} tied"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=6.3, color="white"
                        if r.best_queue in ("CTAS", "SIM-AGE", "SIM") else INK)
        ax.set_xlim(0, 3); ax.set_ylim(len(rows), 0)
        ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels([CFG_LAB[c] for c in CFG])
        ax.set_yticks([i + 0.5 for i in range(len(rows))])
        ax.set_yticklabels([f"{ACU_LAB[a]}, {l}" for a, l in rows])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=QCOL[q]) for q in QUEUES]
    fig.legend(handles, QUEUES, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.55, -0.04))
    fig.savefig(FIG / "fig_best_policy_heatmap.pdf")
    plt.close(fig)


def regret_outputs(rg, best):
    rows = []
    for k, lab in KPIS.items():
        d = rg[rg.kpi == k]
        s = d.groupby(["queue", "routing"]).regret_rel.agg(["mean", "max"]).reset_index()
        s = s.sort_values("mean")
        o = s.iloc[0]
        dq, dr = DEFAULT["queue"], DEFAULT["routing"]
        dd = s[(s.queue == dq) & (s.routing == dr)].iloc[0]
        osfa = d[(d.queue == o.queue) & (d.routing == o.routing)]
        n_opt = int((osfa.regret_rel <= 1e-12).sum())
        rows.append({"kpi": k, "label": lab, "osfa_queue": o.queue, "osfa_routing": o.routing,
                     "osfa_mean": o["mean"], "osfa_max": o["max"], "osfa_n_best": n_opt,
                     "osfa_n_gt10": int((osfa.regret_rel > 0.10).sum()),
                     "default_mean": dd["mean"], "default_max": dd["max"],
                     "default_n_gt10": int((d[(d.queue == dq) & (d.routing == dr)].regret_rel > 0.10).sum()),
                     "osfa_abs_mean": osfa.regret_abs.mean(), "osfa_abs_max": osfa.regret_abs.max(),
                     "worst_type": tlabel(*osfa.loc[osfa.regret_rel.idxmax(), ["acuity", "load", "config"]])})
        key = cname(k)
        macro(f"Osfa{key}Policy", plabel(o.queue, o.routing), "{}")
        macro(f"Osfa{key}MeanRegret", pct(o["mean"]), "{:.1f}")
        macro(f"Osfa{key}MaxRegret", pct(o["max"]), "{:.1f}")
        macro(f"Osfa{key}NBest", n_opt, "{:.0f}")
        macro(f"Osfa{key}NGtTen", rows[-1]["osfa_n_gt10"], "{:.0f}")
        macro(f"Osfa{key}WorstType", rows[-1]["worst_type"], "{}")
        macro(f"Default{key}MeanRegret", pct(dd["mean"]), "{:.1f}")
        macro(f"Default{key}MaxRegret", pct(dd["max"]), "{:.1f}")
        macro(f"Default{key}NGtTen", rows[-1]["default_n_gt10"], "{:.0f}")
        # distinct best policies across the 27 types
        b = best[best.kpi == k]
        macro(f"NDistinctBest{key}", b.groupby(["best_queue", "best_routing"]).ngroups, "{:.0f}")
        macro(f"NDistinctBestQueue{key}", b.best_queue.nunique(), "{:.0f}")
    t = pd.DataFrame(rows)
    t.to_csv(SUPP / "regret_summary.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Best single policy (one size fits all)} & \multicolumn{3}{c}{Default (CTAS/C)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r"KPI & Policy & Mean (\%) & Max (\%) & $>10\%$ & Mean (\%) & Max (\%) & $>10\%$ \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.label} & {plabel(r.osfa_queue, r.osfa_routing)} & {pct(r.osfa_mean):.1f} & {pct(r.osfa_max):.1f} & "
                     f"{r.osfa_n_gt10} & {pct(r.default_mean):.1f} & {pct(r.default_max):.1f} & {r.default_n_gt10} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_regret", "\n".join(lines))
    # cross-KPI regret: policy best for one KPI judged on another
    cross = []
    for k1 in ["ratp", "los_all_mean"]:
        o = t[t.kpi == k1].iloc[0]
        for k2 in ["ratp", "los_all_mean"]:
            d = rg[(rg.kpi == k2) & (rg.queue == o.osfa_queue) & (rg.routing == o.osfa_routing)]
            cross.append((k1, k2, d.regret_rel.mean()))
            macro(f"Cross{cname(k1)}On{cname(k2)}", pct(d.regret_rel.mean()), "{:.1f}")
    # figure: regret per ED type, RATP and LOS, default vs one-size-fits-all
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
    types = [(a, l, c) for c in CFG for a in ACU for l in LOAD]
    x = np.arange(len(types))
    for ax, k in zip(axes, ["ratp", "los_all_mean"]):
        o = t[t.kpi == k].iloc[0]
        d = rg[rg.kpi == k].set_index(["acuity", "load", "config", "queue", "routing"])
        ys_o = [pct(d.loc[(a, l, c, o.osfa_queue, o.osfa_routing), "regret_rel"]) for a, l, c in types]
        ys_d = [pct(d.loc[(a, l, c, DEFAULT["queue"], DEFAULT["routing"]), "regret_rel"]) for a, l, c in types]
        ax.vlines(x - 0.15, 0, ys_d, color=PALETTE[1], lw=2)
        ax.vlines(x + 0.15, 0, ys_o, color=PALETTE[0], lw=2)
        ax.plot(x - 0.15, ys_d, "o", color=PALETTE[1], ms=4, label=f"Default ({plabel(DEFAULT['queue'], DEFAULT['routing'])})")
        ax.plot(x + 0.15, ys_o, "o", color=PALETTE[0], ms=4, label=f"Best single policy ({plabel(o.osfa_queue, o.osfa_routing)})")
        ax.set_ylabel("Regret in RATP (%)" if k == "ratp" else "Regret in mean LOS (%)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7)
        for b in (8.5, 17.5):
            ax.axvline(b, color=MUTED, lw=0.6, ls=":")
        top = ax.get_ylim()[1]
        for i, c in enumerate(CFG):
            ax.text(4 + 9 * i, top * 0.97, CFG_LAB[c], ha="center", va="top", fontsize=7, color=MUTED,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{a[:4].title()}-{l[:3]}" for a, l, c in types], rotation=90, fontsize=6.5)
    fig.savefig(FIG / "fig_regret.pdf")
    plt.close(fig)
    return t


def figure_pareto(m):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    marks = {"": "o", "STREAM-CTAS": "o", "STREAM-PRED": "s", "STREAM-SIM": "^"}
    for ax, c in zip(axes, CFG):
        d = m[(m.acuity == "community") & (m.load == "high") & (m.config == c) & ~m.overloaded]
        pts = d[["los_all_mean", "ratp"]].to_numpy()
        front = [i for i, p in enumerate(pts) if not any((q <= p).all() and (q < p).any() for q in pts)]
        fr = d.iloc[front].sort_values("los_all_mean")
        ax.plot(fr.los_all_mean, fr.ratp, color=MUTED, lw=1, zorder=1)
        for _, r in d.iterrows():
            ax.scatter(r.los_all_mean, r.ratp, s=26, marker=marks[r.routing], color=QCOL[r.queue],
                       edgecolor="white", linewidth=0.8, zorder=2)
        for _, r in fr.iterrows():
            ax.annotate(plabel(r.queue, r.routing), (r.los_all_mean, r.ratp), fontsize=5.6, color=INK,
                        xytext=(3, 2), textcoords="offset points")
        ax.set_title(CFG_LAB[c], fontsize=8, loc="left", color=INK)
        ax.set_xlabel("Mean LOS (min)")
    axes[0].set_ylabel("RATP")
    handles = [plt.Line2D([], [], marker="o", ls="", color=QCOL[q]) for q in QUEUES]
    handles += [plt.Line2D([], [], marker=marks[r], ls="", color=MUTED) for r in ROUTES]
    labels = QUEUES + ["C: CTAS streaming", "P: prediction streaming", "S: similarity routing"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(FIG / "fig_pareto.pdf")
    plt.close(fig)


def figure_tails(e2):
    d = e2[(e2.acuity == "community") & (e2.load == "high") & (e2.config == "POOLED")]
    g = d.groupby("queue")[[f"pia_ctas{l}_p95" for l in range(1, 6)]].mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(1, 6)
    for j, q in enumerate(QUEUES):
        ax.plot(x + (j - 3) * 0.08, g.loc[q].to_numpy() / 60, "o", ms=4, color=QCOL[q], label=q)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"CTAS {l}" for l in x])
    ax.set_ylabel("95th percentile door-to-physician (h)")
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(FIG / "fig_tails.pdf")
    plt.close(fig)
    for q in ("CTAS", "APQ", "SIM", "SIM-AGE", "CTAS-SPT", "FCFS"):
        for l in (2, 4, 5):
            macro(f"TailPNinetyFive{cname(q)}Ctas{cname(str(l))}", g.loc[q, f"pia_ctas{l}_p95"] / 60, "{:.1f}")


def table_kpis_detail(e2, m):
    """Main-text table: community ED, high load, POOLED; queue rules as columns; paired differences vs CTAS."""
    t = ("community", "high", "POOLED")
    rows = [("ratp", "RATP", "{:.3f}"), ("awt", "AWT (min)", "{:.1f}"), ("pia_all_p90", "P90 door to physician (min)", "{:.0f}"),
            ("pia_ctas2_p95", "P95 door to physician, CTAS 2 (min)", "{:.0f}"),
            ("pia_ctas5_p95", "P95 door to physician, CTAS 5 (min)", "{:.0f}"),
            ("los_all_mean", "Mean LOS (min)", "{:.1f}"), ("los_disch_mean", "Mean LOS, discharged (min)", "{:.1f}"),
            ("los_adm_mean", "Mean LOS, admitted (min)", "{:.1f}")]
    lines = [r"\begin{tabular}{l" + "r" * len(QUEUES) + "}", r"\toprule", "KPI & " + " & ".join(QUEUES) + r" \\", r"\midrule"]
    for k, lab, f in rows:
        cells = []
        for q in QUEUES:
            mu = float(e2[(e2.acuity == t[0]) & (e2.load == t[1]) & (e2.config == t[2]) & (e2.queue == q)][k].mean())
            cells.append(f.format(mu))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
        cells = []
        for q in QUEUES:
            if q == "CTAS":
                cells.append("--")
                continue
            dm, hw, _ = paired(e2, t, k, (q, ""), ("CTAS", ""))
            sig = abs(dm) > hw
            cells.append(("\\textbf{" if sig else "") + (f.replace("{:", "{:+") .format(dm)) + ("}" if sig else ""))
        lines.append(r"\quad vs CTAS & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_kpis_community_high_pooled", "\n".join(lines))
    # supplementary: full per-type tables as CSV (all KPIs, mean and half-width)
    g = e2.groupby(TYPE + ["queue", "routing"])
    num = [c for c in e2.columns if e2[c].dtype.kind == "f" and c not in ("alpha",)]
    mean = g[num].mean()
    hw = g[num].std(ddof=1) / np.sqrt(g.size().to_numpy()[:, None]) * T975(g.size().iloc[0])
    out = mean.join(hw, rsuffix="_hw95").reset_index()
    out.to_csv(SUPP / "E2_policy_means_by_ed_type.csv", index=False)


def alpha_outputs():
    e4 = pd.read_csv(RESULTS / "E4_alpha/kpis_alpha.csv.gz")
    e4["routing"] = e4["routing"].fillna("")
    alphas = sorted(e4.alpha.unique())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
    rows = []
    for i, (cfg, k, title) in enumerate([("POOLED", "ratp", "RATP"), ("FASTTRACK", "ratp", "RATP")]):
        for j, a in enumerate(ACU):
            ax = axes[i, j]
            d = e4[(e4.acuity == a) & (e4.config == cfg)]
            if cfg == "POOLED":
                series = [(q, "") for q in ["CTAS-SPT", "CTAS-PD", "SIM", "SIM-AGE"]]
                flat = [("CTAS", ""), ("APQ", "")]
            else:
                series = [("CTAS", "STREAM-PRED"), ("CTAS", "STREAM-SIM"), ("SIM-AGE", "STREAM-SIM"), ("SIM-AGE", "STREAM-CTAS")]
                flat = [("CTAS", "STREAM-CTAS"), ("APQ", "STREAM-CTAS")]
            for q, r in flat:
                v = d[(d.queue == q) & (d.routing == r)][k].mean()
                ax.axhline(v, color=QCOL[q], lw=1, ls="--", label=plabel(q, r) + " (no prediction)")
            for q, r in series:
                ys = [d[(d.queue == q) & (d.routing == r) & (d.alpha == al)][k].mean() for al in alphas]
                mk = {"": "o", "STREAM-CTAS": "o", "STREAM-PRED": "s", "STREAM-SIM": "^"}[r]
                ax.plot(alphas, ys, marker=mk, ms=3.5, lw=1.4, color=QCOL[q], label=plabel(q, r))
            ax.set_title(f"{CFG_LAB[cfg]}, {ACU_LAB[a]}", fontsize=7.5, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel(title)
            if i == 1:
                ax.set_xlabel(r"prediction error level $\alpha$")
            if j == 2:
                ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.18, 1.0))
    fig.savefig(FIG / "fig_alpha.pdf")
    plt.close(fig)
    # break-even alpha: largest alpha at which the best prediction-based policy still beats the best prediction-free one
    for (a, c), d in e4.groupby(["acuity", "config"]):
        for k in ("ratp", "los_all_mean"):
            free = d[~d.uses_prediction].groupby(["queue", "routing"])[k].mean().min()
            pb = d[d.uses_prediction].groupby(["alpha", "queue", "routing"])[k].mean().groupby("alpha").min()
            gain0 = (free - pb.loc[0.0]) / free
            gain1 = (free - pb.loc[1.0]) / free
            better = pb[pb < free]
            be = float(better.index.max()) if len(better) else float("nan")
            rows.append({"acuity": a, "config": c, "kpi": k, "gain_alpha0": gain0, "gain_alpha1": gain1,
                         "max_alpha_still_better": be, "all_alphas_better": len(better) == len(pb)})
    t = pd.DataFrame(rows)
    t.to_csv(SUPP / "E4_value_of_prediction.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{RATP} & \multicolumn{3}{c}{Mean LOS} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Acuity & Config. & Gain $\alpha{=}0$ (\%) & Gain $\alpha{=}1$ (\%) & Max $\alpha$ & Gain $\alpha{=}0$ (\%) & Gain $\alpha{=}1$ (\%) & Max $\alpha$ \\",
             r"\midrule"]
    for a in ACU:
        for c in CFG:
            r1 = t[(t.acuity == a) & (t.config == c) & (t.kpi == "ratp")].iloc[0]
            r2 = t[(t.acuity == a) & (t.config == c) & (t.kpi == "los_all_mean")].iloc[0]
            f = lambda r: ("--" if math.isnan(r.max_alpha_still_better) else
                           (r"$\geq$" + f"{r.max_alpha_still_better:.2f}" if r.all_alphas_better else f"{r.max_alpha_still_better:.2f}"))
            lines.append(f"{ACU_LAB[a] if c == 'POOLED' else ''} & {CFG_LAB[c]} & {pct(r1.gain_alpha0):.1f} & {pct(r1.gain_alpha1):.1f} & {f(r1)} & "
                         f"{pct(r2.gain_alpha0):.1f} & {pct(r2.gain_alpha1):.1f} & {f(r2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_value_of_prediction", "\n".join(lines))
    for k in ("ratp", "los_all_mean"):
        d = t[t.kpi == k]
        macro(f"VopGainZeroMax{cname(k)}", pct(d.gain_alpha0.max()), "{:.1f}")
        macro(f"VopGainOneMax{cname(k)}", pct(d.gain_alpha1.max()), "{:.1f}")
        macro(f"VopGainOneMin{cname(k)}", pct(d.gain_alpha1.min()), "{:.1f}")
        macro(f"VopNBetterAtOne{cname(k)}", int((d.gain_alpha1 > 0).sum()), "{:.0f}")
        macro(f"VopNAllAlpha{cname(k)}", int(d.all_alphas_better.sum()), "{:.0f}")
        macro(f"VopGainDrop{cname(k)}", pct((d.gain_alpha0 - d.gain_alpha1).mean()), "{:.1f}")
    return t


def sensitivity_outputs(best_core):
    e5 = pd.read_csv(RESULTS / "E5_sensitivity/kpis_sensitivity.csv.gz")
    e5["routing"] = e5["routing"].fillna("")
    rows = []
    for (b, dg), dv in e5.groupby(["boarding", "diagnostics"]):
        m = dv.groupby(TYPE + ["queue", "routing"])[list(KPIS)].mean().reset_index()
        for k in ("ratp", "los_all_mean"):
            for (a, l, c), g in m.groupby(TYPE):
                i = g[k].idxmin()
                core = best_core[(best_core.acuity == a) & (best_core.load == l) & (best_core.config == c) & (best_core.kpi == k)].iloc[0]
                v_core = float(g[(g.queue == core.best_queue) & (g.routing == core.best_routing)][k].iloc[0])
                rows.append({"variant": f"{b}|{dg}", "kpi": k, "acuity": a, "config": c,
                             "best_queue": g.loc[i, "queue"], "best_routing": g.loc[i, "routing"],
                             "same_as_core": (g.loc[i, "queue"], g.loc[i, "routing"]) == (core.best_queue, core.best_routing),
                             "regret_core_best": v_core / float(g.loc[i, k]) - 1})
    t = pd.DataFrame(rows)
    t.to_csv(SUPP / "E5_sensitivity_best.csv", index=False)
    vlab = {"high|typical": "High boarding", "low|light": "Light diagnostics", "low|heavy": "Heavy diagnostics"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{RATP} & \multicolumn{2}{c}{Mean LOS} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Variant & Same best & Regret of core best (\%) & Same best & Regret of core best (\%) \\", r"\midrule"]
    for v in ["high|typical", "low|light", "low|heavy"]:
        cells = []
        for k in ("ratp", "los_all_mean"):
            d = t[(t.variant == v) & (t.kpi == k)]
            cells += [f"{int(d.same_as_core.sum())}/{len(d)}", f"{pct(d.regret_core_best.mean()):.1f} (max {pct(d.regret_core_best.max()):.1f})"]
            macro(f"Sens{cname(vlab[v])}{cname(k)}Same", int(d.same_as_core.sum()), "{:.0f}")
            macro(f"Sens{cname(vlab[v])}{cname(k)}Regret", pct(d.regret_core_best.mean()), "{:.1f}")
            macro(f"Sens{cname(vlab[v])}{cname(k)}RegretMax", pct(d.regret_core_best.max()), "{:.1f}")
        lines.append(vlab[v] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_sensitivity", "\n".join(lines))
    # ED types near saturation in a variant: median Main-space utilisation over policies >= 0.95
    for (b, dg), dv in e5.groupby(["boarding", "diagnostics"]):
        u = dv.groupby(TYPE + ["queue", "routing"]).util_MAIN.mean().groupby(TYPE).median()
        near = u[u >= 0.95]
        nm = cname(vlab[f"{b}|{dg}"])
        macro(f"Sens{nm}NearSat", len(near), "{:.0f}")
        macro(f"Sens{nm}NearSatList", "; ".join(tlabel(*k) for k in near.index) if len(near) else "none", "{}")
        macro(f"Sens{nm}UtilMax", float(u.max()), "{:.2f}")
    # does the best queue rule under high boarding shift? count by queue
    hb = t[(t.variant == "high|typical") & (t.kpi == "ratp")]
    macro("SensHighBoardingBestQueues", " and ".join(sorted(hb.best_queue.unique())), "{}")
    return t


def decision_tree(best):
    feats = []
    labels = {}
    for k in ("ratp", "los_all_mean"):
        b = best[best.kpi == k].copy()
        X = pd.DataFrame({"acuity": b.acuity.map({"urgentcare": 0, "community": 1, "tertiary": 2}),
                          "load": b.load.map({"low": 0, "medium": 1, "high": 2}),
                          "fasttrack": (b.config == "FASTTRACK").astype(int),
                          "splitflow": (b.config == "SPLITFLOW").astype(int)})
        y = b.best_queue.to_numpy()
        clf = DecisionTreeClassifier(max_depth=3, random_state=0, min_samples_leaf=2).fit(X, y)
        acc = clf.score(X, y)
        txt = export_text(clf, feature_names=list(X.columns))
        (SUPP / f"decision_tree_{k}.txt").write_text(txt)
        macro(f"Tree{cname(k)}Acc", pct(acc), "{:.0f}")
        macro(f"Tree{cname(k)}Leaves", clf.get_n_leaves(), "{:.0f}")
        labels[k] = (clf, X, y)


def decision_guide(rg):
    """For each acuity mix and configuration: the policy with the smallest mean regret over the
    three load levels, for RATP, for mean LOS, and for their average (balanced)."""
    w = rg[rg.kpi.isin(["ratp", "los_all_mean"])].pivot_table(
        index=["acuity", "load", "config", "queue", "routing"], columns="kpi", values="regret_rel").reset_index()
    w["balanced"] = (w.ratp + w.los_all_mean) / 2
    rows = []
    for a in ACU:
        for c in CFG:
            d = w[(w.acuity == a) & (w.config == c)]
            if c == "POOLED":
                d = d[d.routing == ROUTES[0]]      # routing is ignored in pooled EDs
            r = {"acuity": a, "config": c}
            for k in ("ratp", "los_all_mean", "balanced"):
                s_ = d.groupby(["queue", "routing"])[k].agg(["mean", "max"]).sort_values("mean")
                (q, rt), v = s_.index[0], s_.iloc[0]
                r[k] = (plabel(q, "" if c == "POOLED" else rt), v["mean"], v["max"])
            rows.append(r)
    lines = [r"\begin{tabular}{llrlrlr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Access (RATP)} & \multicolumn{2}{c}{Flow (mean LOS)} & \multicolumn{1}{c}{Balanced} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
             r"Acuity mix & Configuration & Policy & Max regret (\%) & Policy & Max regret (\%) & Policy (max regret, \%) \\", r"\midrule"]
    for r in rows:
        b = r["balanced"]
        lines.append(f"{ACU_LAB[r['acuity']] if r['config'] == 'POOLED' else ''} & {CFG_LAB[r['config']]} & "
                     f"{r['ratp'][0]} & {pct(r['ratp'][2]):.1f} & {r['los_all_mean'][0]} & {pct(r['los_all_mean'][2]):.1f} & "
                     f"{b[0]} ({pct(max(b[2], 0)):.1f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("tab_decision_guide", "\n".join(lines))
    macro("GuideNDistinctBalanced", len({r["balanced"][0] for r in rows}), "{:.0f}")
    macro("GuideMaxRegretRatp", pct(max(r["ratp"][2] for r in rows)), "{:.1f}")
    macro("GuideMaxRegretLos", pct(max(r["los_all_mean"][2] for r in rows)), "{:.1f}")
    pd.DataFrame([{**{"acuity": r["acuity"], "config": r["config"]},
                   **{f"{k}_{x}": r[k][i] for k in ("ratp", "los_all_mean", "balanced") for i, x in enumerate(("policy", "mean_regret", "max_regret"))}}
                  for r in rows]).to_csv(SUPP / "decision_guide.csv", index=False)


def headline_macros(best, m, e2):
    for k in ("ratp", "los_all_mean"):
        b = best[best.kpi == k]
        rel = (b.default - b.best) / b.default
        macro(f"BestVsDefault{cname(k)}Mean", pct(rel.mean()), "{:.1f}")
        macro(f"BestVsDefault{cname(k)}Max", pct(rel.max()), "{:.1f}")
        macro(f"BestVsDefault{cname(k)}Min", pct(rel.min()), "{:.1f}")
        sig = (b.best_minus_default + b.best_minus_default_hw < 0)
        macro(f"BestVsDefault{cname(k)}NSig", int(sig.sum()), "{:.0f}")
        macro(f"DefaultIsBest{cname(k)}", int(((b.best_queue == DEFAULT['queue']) & ((b.best_routing == DEFAULT['routing']) | (b.config == 'POOLED'))).sum()), "{:.0f}")
        cnt = b.best_queue.value_counts()
        for q in QUEUES:
            macro(f"NBest{cname(k)}{cname(q)}", int(cnt.get(q, 0)), "{:.0f}")
        rc = b[b.config != "POOLED"].best_routing.value_counts()
        for r in ROUTES:
            macro(f"NBestRoute{cname(k)}{cname(r)}", int(rc.get(r, 0)), "{:.0f}")
        macro(f"MedianTied{cname(k)}", float(b.n_tied.median()), "{:.0f}")
    macro("NEdTypes", best[["acuity", "load", "config"]].drop_duplicates().shape[0], "{:.0f}")
    macro("NPoliciesTwoArea", len(candidates_for("FASTTRACK")), "{:.0f}")
    macro("NQueueRules", len(QUEUES), "{:.0f}")
    macro("NRuns", len(e2), "{:,.0f}")
    macro("PatientsPerRunMean", e2.n_patients.mean(), "{:,.0f}")
    # configuration effect under the default policy (community, high load)
    for c in CFG:
        v = m[(m.acuity == "community") & (m.load == "high") & (m.config == c) & (m.queue == "CTAS") &
              (m.routing == ("" if c == "POOLED" else "STREAM-CTAS"))]
        macro(f"CommHighDefaultLos{cname(c)}", float(v.los_all_mean.iloc[0]), "{:.0f}")
        macro(f"CommHighDefaultRatp{cname(c)}", float(v.ratp.iloc[0]), "{:.2f}")
    # load effect on RATP of best policies
    for l in LOAD:
        b = best[(best.kpi == "ratp") & (best.load == l)]
        macro(f"BestRatpGain{cname(l)}", pct(((b.default - b.best) / b.default).mean()), "{:.1f}")
    # FCFS and prediction streaming as worst cases
    worst = m.loc[m.groupby(TYPE).ratp.idxmax()]
    macro("NWorstRatpFcfs", int((worst.queue == "FCFS").sum()), "{:.0f}")
    two = m[m.config != "POOLED"]
    worst2 = two.loc[two.groupby(TYPE).los_all_mean.idxmax()]
    macro("NWorstLosPredStream", int((worst2.routing == "STREAM-PRED").sum()), "{:.0f}")
    macro("NTwoAreaTypes", two[TYPE].drop_duplicates().shape[0], "{:.0f}")
    macro("NMediumTypes", m[m.load == P.design()["alpha_ed_types_load"]][TYPE].drop_duplicates().shape[0], "{:.0f}")
    ov = m[m.overloaded]
    macro("NOverloadedCombos", len(ov), "{:.0f}")
    macro("NOverloadedPred", int((ov.routing == "STREAM-PRED").sum()), "{:.0f}")
    macro("NOverloadedTypes", ov[TYPE].drop_duplicates().shape[0], "{:.0f}")
    macro("OverloadedTypesList", "; ".join(sorted({tlabel(a, l, c) for a, l, c in ov[TYPE].itertuples(index=False)})), "{}")
    macro("OverloadThreshold", OVERLOAD)
    for r in ROUTES:
        d = two[two.routing == r]
        macro(f"ShareSecond{cname(r)}Min", pct(d.share_second_area.min()), "{:.0f}")
        macro(f"ShareSecond{cname(r)}Max", pct(d.share_second_area.max()), "{:.0f}")
    q = two[two.queue == "CTAS"].set_index(TYPE + ["routing"])
    cmp = []
    for t in two[TYPE].drop_duplicates().itertuples(index=False):
        a = q.loc[tuple(t) + ("STREAM-SIM",)]; b = q.loc[tuple(t) + ("STREAM-CTAS",)]
        cmp.append((a.pia_ctas3_mean / b.pia_ctas3_mean - 1, a.pia_ctas4_mean / b.pia_ctas4_mean - 1, a.ratp / b.ratp - 1, a.los_all_mean / b.los_all_mean - 1))
    cmp = np.array(cmp)
    macro("SimVsCtasRouteCtasThreeWait", -pct(cmp[:, 0].mean()), "{:.0f}")
    macro("SimVsCtasRouteCtasFourWait", pct(cmp[:, 1].mean()), "{:.0f}")
    macro("SimVsCtasRouteRatp", -pct(cmp[:, 2].mean()), "{:.1f}")
    macro("SimVsCtasRouteNBetterRatp", int((cmp[:, 2] < 0).sum()), "{:.0f}")
    macro("SimVsCtasRouteNBetterLos", int((cmp[:, 3] < 0).sum()), "{:.0f}")


def main():
    r = subprocess.run([sys.executable, str(ROOT / "experiments/scripts/check_results.py")], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit("checks failed; outputs not built")
    e2 = load()
    m = policy_table(e2)
    table_parameters()
    table_capacity()
    table_validation()
    table_fidelity()
    best = best_by_type(e2, m)
    best.to_csv(SUPP / "E2_best_policy_by_ed_type.csv", index=False)
    table_best(best)
    figure_heatmap(best)
    rg = regret(m)
    rg.to_csv(SUPP / "E2_regret_by_policy_and_ed_type.csv", index=False)
    regret_outputs(rg, best)
    figure_pareto(m)
    figure_tails(e2)
    table_kpis_detail(e2, m)
    if (RESULTS / "E4_alpha/kpis_alpha.csv.gz").exists():
        alpha_outputs()
    else:
        print("WARNING: E4 missing; alpha outputs skipped")
    if (RESULTS / "E5_sensitivity/kpis_sensitivity.csv.gz").exists():
        sensitivity_outputs(best)
    else:
        print("WARNING: E5 missing; sensitivity outputs skipped")
    decision_tree(best)
    decision_guide(rg)
    headline_macros(best, m, e2)
    import outputs_appendix as oa          # appendix tables/figures share this module's registry
    oa.mo = sys.modules[__name__]
    oa.build_all()
    if (RESULTS / "E6_optimization/opt_test_kpis.csv.gz").exists():
        import outputs_v11 as ov          # revision experiments E6-E9, bootstrap, alternative weights
        ov.mo = sys.modules[__name__]
        ov.build_all(e2)
    else:
        print("WARNING: E6 missing; revision outputs skipped")
    prov = provenance.record("outputs")
    srcs = {}
    for f in sorted(RESULTS.glob("*/*.provenance.json")):
        p = json.loads(f.read_text())
        srcs[f"{f.parent.name}/{p.get('file')}"] = {k: p.get(k) for k in ("git_commit", "config_sha256_16", "date_utc", "replications")}
    prov["sources"] = srcs
    (PAPER / "outputs_provenance.json").write_text(json.dumps(prov, indent=2))
    commits = sorted({v["git_commit"] for v in srcs.values()})
    lines = ["% Generated by experiments/scripts/make_outputs.py -- do not edit by hand.",
             f"% Results commit(s): {', '.join(commits)}; config {prov['config_sha256_16']}; built {prov['date_utc']}"]
    for k in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
    lines.append(f"\\newcommand{{\\ResultsCommit}}{{{commits[0][:7] if commits else 'unknown'}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(MACROS)} macros, tables to {TAB}, figures to {FIG}")


if __name__ == "__main__":
    main()
