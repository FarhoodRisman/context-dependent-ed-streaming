"""Outputs of the revision experiments: optimization (E6), transfer (E7), robustness (E8),
out-of-sample ED types (E9), bootstrap stability and alternative acuity weights (E2).
Called by make_outputs.py; uses its macro registry and writers (module attribute `mo`)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

mo = None   # set by make_outputs

TYPE = ["acuity", "load", "config"]
OBJ = ["access", "balanced"]
SHORT_CFG = {"POOLED": "Pooled", "FASTTRACK": "Fast track", "SPLITFLOW": "Split flow"}
SHORT_ACU = {"tertiary": "tert.", "community": "comm.", "urgentcare": "urg."}
OBJ_LAB = {"access": "Access (RATP)", "balanced": "Balanced"}


def _P():
    return mo.P


def objective_frame(df, default_df, objective):
    """Objective value per row; balanced uses the default policy's RATP and LOS on the same replication."""
    if objective == "access":
        return df["ratp"]
    ref = default_df.set_index(TYPE + ["rep"])[["ratp", "los_all_mean"]]
    k = df.set_index(TYPE + ["rep"]).index
    r = ref.reindex(k)
    return pd.Series(0.5 * df["ratp"].to_numpy() / r["ratp"].to_numpy() +
                     0.5 * df["los_all_mean"].to_numpy() / r["los_all_mean"].to_numpy(), index=df.index)


def default_rows(e2):
    pc = _P().policies_config()["default_policy"]
    return e2[(e2.queue == pc["queue"]) & ((e2.routing == pc["routing"]) | (e2.config == "POOLED"))]


# ------------------------------------------------------------------ E6 optimization
def optimization_outputs(e2):
    R = mo.RESULTS
    opt = pd.read_csv(R / "E6_optimization/opt_test_kpis.csv.gz")
    opt["routing"] = opt["routing"].fillna("")
    specs = json.loads((R / "E6_optimization/opt_specs.json").read_text())
    hist = pd.read_csv(R / "E6_optimization/opt_history.csv.gz")
    prov = json.loads((R / "E6_optimization/opt_test_kpis.provenance.json").read_text())
    cem = prov["cem"]
    mo.macro("CemIter", cem["iterations"], "{:.0f}"); mo.macro("CemPop", cem["population"], "{:.0f}")
    mo.macro("CemElite", cem["elite"], "{:.0f}"); mo.macro("CemSmooth", cem["smoothing"])
    mo.macro("CemShort", cem["shortlist"], "{:.0f}")
    mo.macro("NTrainReps", len(prov["train_reps"]), "{:.0f}"); mo.macro("NValReps", len(prov["validation_reps"]), "{:.0f}")
    mo.macro("CemEvalsPerType", np.mean([s["evaluations"] for s in specs]), "{:,.0f}")
    dflt = default_rows(e2)
    reps = sorted(opt.rep.unique())
    e2t = e2[e2.rep.isin(reps)]
    rows = []
    for o in OBJ:
        fixed = e2t.copy()
        fixed["J"] = objective_frame(fixed, dflt, o)
        oo = opt[opt.objective == o].copy()
        oo["J"] = objective_frame(oo, dflt, o)
        dj = dflt[dflt.rep.isin(reps)].copy()
        dj["J"] = objective_frame(dj, dflt, o)
        for (a, l, c), g in fixed.groupby(TYPE):
            mean = g.groupby(["queue", "routing"]).J.mean()
            bq, br = mean.idxmin()
            jb = g[(g.queue == bq) & (g.routing == br)].set_index("rep").J
            jo = oo[(oo.acuity == a) & (oo.load == l) & (oo.config == c)].set_index("rep").J
            jd = dj[(dj.acuity == a) & (dj.load == l) & (dj.config == c)].set_index("rep").J
            d = (jo - jb).dropna()
            n = len(d)
            hw = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
            ko = oo[(oo.acuity == a) & (oo.load == l) & (oo.config == c)]
            kb = g[(g.queue == bq) & (g.routing == br)]
            rows.append({"objective": o, "acuity": a, "load": l, "config": c, "best_fixed": mo.plabel(bq, br),
                         "J_default": jd.mean(), "J_best_fixed": jb.mean(), "J_opt": jo.mean(),
                         "gain_vs_default": 1 - jo.mean() / jd.mean(), "gain_vs_best": 1 - jo.mean() / jb.mean(),
                         "diff_hw": hw, "sig_better": (d.mean() + hw) < 0, "sig_worse": (d.mean() - hw) > 0,
                         "ratp_opt": ko.ratp.mean(), "los_opt": ko.los_all_mean.mean(),
                         "ratp_best": kb.ratp.mean(), "los_best": kb.los_all_mean.mean(),
                         "p95c5_opt": ko.pia_ctas5_p95.mean(), "p95c5_best": kb.pia_ctas5_p95.mean(),
                         "p95c2_opt": ko.pia_ctas2_p95.mean(), "p95c2_best": kb.pia_ctas2_p95.mean(),
                         "share2_opt": ko.share_second_area.mean(), "share2_best": kb.share_second_area.mean(),
                         "share2_default": dflt[(dflt.acuity == a) & (dflt.load == l) & (dflt.config == c) & dflt.rep.isin(reps)].share_second_area.mean(),
                         "losadm_opt": ko.los_adm_mean.mean(), "losadm_best": kb.los_adm_mean.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(mo.SUPP / "E6_optimized_vs_fixed.csv", index=False)
    for o in OBJ:
        d = t[t.objective == o]
        k = mo.cname(o)
        mo.macro(f"Opt{k}GainBestMean", mo.pct(d.gain_vs_best.mean()), "{:.1f}")
        mo.macro(f"Opt{k}GainBestMax", mo.pct(d.gain_vs_best.max()), "{:.1f}")
        mo.macro(f"Opt{k}GainBestMin", mo.pct(d.gain_vs_best.min()), "{:.1f}")
        mo.macro(f"Opt{k}GainDefaultMean", mo.pct(d.gain_vs_default.mean()), "{:.1f}")
        mo.macro(f"Opt{k}GainDefaultMax", mo.pct(d.gain_vs_default.max()), "{:.1f}")
        mo.macro(f"Opt{k}NSigBetter", int(d.sig_better.sum()), "{:.0f}")
        mo.macro(f"Opt{k}NSigWorse", int(d.sig_worse.sum()), "{:.0f}")
        for cfg in mo.CFG:
            mo.macro(f"Opt{k}GainBest{mo.cname(cfg)}", mo.pct(d[d.config == cfg].gain_vs_best.mean()), "{:.1f}")
        lc = d.los_opt / d.los_best - 1
        mo.macro(f"Opt{k}LosChangeMedian", mo.pct(lc.median()), "{:.1f}")
        mo.macro(f"Opt{k}LosChangeMax", mo.pct(lc.max()), "{:.0f}")
        mo.macro(f"Opt{k}NLosWorseFive", int((lc > 0.05).sum()), "{:.0f}")
        sf = d[d.config == "SPLITFLOW"]
        mo.macro(f"Opt{k}ShareSecondSplitOpt", mo.pct(sf.share2_opt.mean()), "{:.0f}")
        mo.macro(f"Opt{k}ShareSecondSplitBest", mo.pct(sf.share2_best.mean()), "{:.0f}")
        mo.macro(f"Opt{k}ShareSecondSplitDefault", mo.pct(sf.share2_default.mean()), "{:.0f}")
        ft = d[d.config == "FASTTRACK"]
        mo.macro(f"Opt{k}ShareSecondFastOpt", mo.pct(ft.share2_opt.mean()), "{:.0f}")
        mo.macro(f"Opt{k}ShareSecondFastBest", mo.pct(ft.share2_best.mean()), "{:.0f}")
        mo.macro(f"Opt{k}NUrgPooledWorse", int(((d.config == "POOLED") & (d.acuity == "urgentcare") & (d.gain_vs_best < 0)).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NNegGain", int((d.gain_vs_best < 0).sum()), "{:.0f}")
        mo.macro(f"Opt{k}CtasFiveTailRatio", float((d.p95c5_opt / d.p95c5_best).median()), "{:.2f}")
        mo.macro(f"Opt{k}CtasTwoTailRatio", float((d.p95c2_opt / d.p95c2_best).median()), "{:.2f}")
    # table: summary by acuity x configuration (mean over load levels) for both objectives
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Access objective (RATP)} & \multicolumn{3}{c}{Balanced objective} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Acuity mix & Configuration & vs default & vs best fixed & Better/worse & vs default & vs best fixed & Better/worse \\",
             r"\midrule"]
    for a in mo.ACU:
        for c in mo.CFG:
            cells = []
            for o in OBJ:
                d = t[(t.objective == o) & (t.acuity == a) & (t.config == c)]
                cells += [f"{mo.pct(d.gain_vs_default.mean()):.1f}", f"{mo.pct(d.gain_vs_best.mean()):.1f}",
                          f"{int(d.sig_better.sum())}/{int(d.sig_worse.sum())}"]
            lines.append(f"{mo.ACU_LAB[a] if c == 'POOLED' else ''} & {mo.CFG_LAB[c]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_opt_summary", "\n".join(lines))
    # full table (appendix)
    lines = [r"\begin{tabular}{lllrrrlrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Access (RATP)} & \multicolumn{4}{c}{Balanced ($J$)} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-10}",
             r"Acuity & Load & Config. & Default & Best fixed & Optimized & Best fixed policy & Default & Best fixed & Optimized \\",
             r"\midrule"]
    for a in mo.ACU:
        for l in mo.LOAD:
            for c in mo.CFG:
                x = t[(t.objective == "access") & (t.acuity == a) & (t.load == l) & (t.config == c)].iloc[0]
                y = t[(t.objective == "balanced") & (t.acuity == a) & (t.load == l) & (t.config == c)].iloc[0]
                star = lambda r: r"$^{*}$" if r.sig_better else (r"$^{\dagger}$" if r.sig_worse else "")
                lines.append(f"{mo.ACU_LAB[a] if (l == 'low' and c == 'POOLED') else ''} & {l if c == 'POOLED' else ''} & {mo.CFG_LAB[c]} & "
                             f"{x.J_default:.3f} & {x.J_best_fixed:.3f} & {x.J_opt:.3f}{star(x)} & {y.best_fixed} & "
                             f"{y.J_default:.3f} & {y.J_best_fixed:.3f} & {y.J_opt:.3f}{star(y)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_opt_full", "\n".join(lines))
    figure_opt_gain(t)
    structure_outputs(specs)
    convergence_outputs(hist, specs, t)
    return t, specs


def figure_opt_gain(t):
    fig, axes = mo.plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    types = [(a, l, c) for c in mo.CFG for a in mo.ACU for l in mo.LOAD]
    y = np.arange(len(types))[::-1]
    for ax, o in zip(axes, OBJ):
        d = t[t.objective == o].set_index(TYPE)
        bf = [mo.pct(1 - d.loc[k].J_best_fixed / d.loc[k].J_default) for k in types]
        op = [mo.pct(d.loc[k].gain_vs_default) for k in types]
        for yy, b, p in zip(y, bf, op):
            ax.plot([b, p], [yy, yy], color=mo.GRID, lw=1.2, zorder=1)
        ax.scatter(bf, y, s=18, color=mo.PALETTE[1], label="Best fixed policy", zorder=2)
        ax.scatter(op, y, s=18, color=mo.PALETTE[0], label="Optimized similarity policy", zorder=3)
        ax.axvline(0, color=mo.MUTED, lw=0.8)
        for b in (8.5, 17.5):
            ax.axhline(b, color=mo.MUTED, lw=0.6, ls=":")
        ax.set_xlabel("Improvement over the default policy (%)")
        ax.set_title(f"({'ab'[OBJ.index(o)]}) {OBJ_LAB[o]} objective", fontsize=8, loc="left", color=mo.INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{SHORT_CFG[c]}, {SHORT_ACU[a]}, {l}" for a, l, c in types], fontsize=6)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(mo.FIG / "fig_opt_gain.pdf")
    mo.plt.close(fig)


def structure_outputs(specs):
    """What do the optimized rules look like? Weights, targets, ageing, reassessment bonus, routing."""
    rows = []
    for s in specs:
        q, r = s["queue"], s["routing"]
        w = q["weights"]
        rows.append({"objective": s["objective"], "acuity": s["acuity"], "load": s["load"], "config": s["config"],
                     **{f"w{k}": x for k, x in zip("ABCDEF", w)}, **{f"q{k}": x for k, x in zip("ABCDEF", q["target"])},
                     "gamma": q["aging"], "beta": q["ra_bonus"],
                     "lex_A": w[0] / 4 > sum(w[1:]),
                     **({f"v{k}": x for k, x in zip("ABCDEF", r["weights"])} if r else {}),
                     "offset": r["offset"] if r else np.nan, "J_train": s["J_train"], "J_val": s["J_validation"]})
    d = pd.DataFrame(rows)
    d.to_csv(mo.SUPP / "E6_optimized_parameters.csv", index=False)
    for o in OBJ:
        x = d[d.objective == o]
        k = mo.cname(o)
        mo.macro(f"Opt{k}NLexA", int(x.lex_A.sum()), "{:.0f}")
        mo.macro(f"Opt{k}NWbGtWa", int((x.wB > x.wA).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NWbGtWaHigh", int((x[x.load == "high"].wB > x[x.load == "high"].wA).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NWbGtWaLow", int((x[x.load == "low"].wB > x[x.load == "low"].wA).sum()), "{:.0f}")
        mo.macro(f"Opt{k}WbMedianHigh", float(x[x.load == "high"].wB.median()))
        mo.macro(f"Opt{k}WbMedianLow", float(x[x.load == "low"].wB.median()))
        mo.macro(f"Opt{k}NBetaPosHigh", int((x[x.load == "high"].beta > 0).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NBetaNegLowMed", int((x[x.load != "high"].beta < 0).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NLowMed", int((x.load != "high").sum()), "{:.0f}")
        mo.macro(f"Opt{k}GammaMax", float(x.gamma.max()), "{:.3f}")
        mo.macro(f"Opt{k}NqCDisch", int((x.qC == 2).sum()), "{:.0f}")
        mo.macro(f"Opt{k}WAMedian", float(x.wA.median()))
        mo.macro(f"Opt{k}NBetaPos", int((x.beta > 0.05).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NBetaNeg", int((x.beta < -0.05).sum()), "{:.0f}")
        mo.macro(f"Opt{k}BetaMedian", float(x.beta.median()))
        mo.macro(f"Opt{k}GammaMedian", float(x.gamma.median()), "{:.3f}")
        mo.macro(f"Opt{k}NGammaSmall", int((x.gamma < 0.02).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NqBFive", int((x.qB == 5).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NqBInterior", int(((x.qB > 1) & (x.qB < 5)).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NqCTwo", int((x.qC == 2).sum()), "{:.0f}")
        two = x[x.config != "POOLED"]
        mo.macro(f"Opt{k}OffsetMedian", float(two.offset.median()))
        mo.macro(f"Opt{k}NOffsetNeg", int((two.offset < 0).sum()), "{:.0f}")
        mo.macro(f"Opt{k}NTwoArea", len(two), "{:.0f}")
    # heatmap of parameters (balanced objective in main text; access in appendix)
    for o in OBJ:
        x = d[d.objective == o].copy()
        x["order"] = x.apply(lambda r: (mo.CFG.index(r.config), mo.ACU.index(r.acuity), mo.LOAD.index(r.load)), axis=1)
        x = x.sort_values("order")
        cols = ["wA", "wB", "wC", "wD", "wE", "wF", "gamma", "beta", "vA", "vB", "vC", "vD", "vE", "vF", "offset"]
        labs = ["$w_A$", "$w_B$", "$w_C$", "$w_D$", "$w_E$", "$w_F$", r"$\gamma$", r"$\beta$",
                "$v_A$", "$v_B$", "$v_C$", "$v_D$", "$v_E$", "$v_F$", "$b$"]
        M = x[cols].to_numpy(dtype=float)
        fig, ax = mo.plt.subplots(figsize=(7.0, 5.8))
        from matplotlib.colors import TwoSlopeNorm
        cmap = mo.plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if (not np.isnan(v) and (v > 0.6 or v < -0.35)) else mo.INK)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=8.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(x)))
        ax.set_yticklabels([f"{mo.CFG_LAB[r.config]}, {mo.ACU_LAB[r.acuity]}, {r.load}  [{'-'.join(str(int(r[f'q{k}'])) for k in 'ABCDEF')}]"
                            for _, r in x.iterrows()], fontsize=6.5)
        ax.axvline(7.5, color="white", lw=2); ax.axvline(5.5, color="white", lw=1)
        for b in (8.5, 17.5):
            ax.axhline(b, color="white", lw=2)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        ax.tick_params(length=0)
        fig.savefig(mo.FIG / f"fig_opt_params_{o}.pdf")
        mo.plt.close(fig)


def convergence_outputs(hist, specs, t):
    fig, axes = mo.plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, o in zip(axes, OBJ):
        h = hist[hist.objective == o]
        for _, g in h.groupby(TYPE):
            g = g.sort_values("iteration")
            ax.plot(g.iteration + 1, g.best_J / g.best_J.iloc[0], color=mo.GRID, lw=0.8)
        med = h.assign(rel=h.groupby(TYPE).best_J.transform(lambda s: s / s.iloc[0])).groupby("iteration").rel.median()
        ax.plot(med.index + 1, med.values, color=mo.PALETTE[0], lw=2, label="median over ED types")
        ax.set_xlabel("Iteration")
        ax.set_title(f"({'ab'[OBJ.index(o)]}) {OBJ_LAB[o]} objective", fontsize=8, loc="left", color=mo.INK)
    axes[0].set_ylabel("Best training objective\n(relative to iteration 1)")
    axes[0].legend(fontsize=6.5)
    fig.savefig(mo.FIG / "fig_cem_convergence.pdf")
    mo.plt.close(fig)
    # optimism: training/validation objective versus test objective of the selected policy
    sp = pd.DataFrame([{"objective": s["objective"], "acuity": s["acuity"], "load": s["load"], "config": s["config"],
                        "J_train": s["J_train"], "J_val": s["J_validation"]} for s in specs])
    m = sp.merge(t, on=["objective"] + TYPE)
    for o in OBJ:
        x = m[m.objective == o]
        mo.macro(f"Opt{mo.cname(o)}TrainOptimism", mo.pct(((x.J_train - x.J_opt) / x.J_opt).mean()), "{:.1f}")
        mo.macro(f"Opt{mo.cname(o)}ValOptimism", mo.pct(((x.J_val - x.J_opt) / x.J_opt).mean()), "{:.1f}")


# ------------------------------------------------------------------ E7 transfer
def transfer_outputs():
    d = pd.read_csv(mo.RESULTS / "E7_transfer/transfer_kpis.csv.gz")
    mo.macro("NTransferReps", d.rep.nunique(), "{:.0f}")
    mo.macro("TransferBigRegret", 50, "{:.0f}")
    dflt = d[d.source == "DEFAULT"].set_index(TYPE + ["rep"])
    rows = []
    for o in OBJ:
        x = d[(d.objective == o)].copy()
        if o == "access":
            x["J"] = x.ratp
        else:
            r = dflt.reindex(x.set_index(TYPE + ["rep"]).index)
            x["J"] = 0.5 * x.ratp.to_numpy() / r.ratp.to_numpy() + 0.5 * x.los_all_mean.to_numpy() / r.los_all_mean.to_numpy()
        m = x.groupby(TYPE + ["source"]).J.mean().reset_index()
        for (a, l, c), g in m.groupby(TYPE):
            own = g[g.source == f"{a}|{l}"].J.iloc[0]
            for _, rr in g.iterrows():
                rows.append({"objective": o, "config": c, "target": f"{a}|{l}", "source": rr.source,
                             "regret": rr.J / own - 1})
    t = pd.DataFrame(rows)
    t.to_csv(mo.SUPP / "E7_transfer_regret.csv", index=False)
    order = [f"{a}|{l}" for a in mo.ACU for l in mo.LOAD]
    lab = [f"{a[:4].title()}-{l[:3]}" for a in mo.ACU for l in mo.LOAD]
    fig, axes = mo.plt.subplots(2, 3, figsize=(7.0, 4.9))
    for i, o in enumerate(OBJ):
        for j, c in enumerate(mo.CFG):
            ax = axes[i, j]
            M = t[(t.objective == o) & (t.config == c)].pivot(index="source", columns="target", values="regret").reindex(index=order, columns=order)
            A = mo.pct(M.to_numpy())
            ax.imshow(np.clip(A, 0, 30), cmap="Blues", vmin=0, vmax=30, aspect="equal")
            for r_ in range(9):
                for c_ in range(9):
                    ax.text(c_, r_, f"{A[r_, c_]:.0f}", ha="center", va="center", fontsize=4.6,
                            color="white" if A[r_, c_] > 18 else mo.INK)
            ax.set_xticks(range(9)); ax.set_yticks(range(9))
            ax.set_xticklabels(lab, rotation=90, fontsize=5); ax.set_yticklabels(lab, fontsize=5)
            ax.set_title(f"{OBJ_LAB[o]}, {mo.CFG_LAB[c]}", fontsize=7, loc="left", color=mo.INK)
            if j == 0:
                ax.set_ylabel("Optimized for (source)", fontsize=6)
            if i == 1:
                ax.set_xlabel("Applied to (target)", fontsize=6)
            ax.tick_params(length=0)
    fig.savefig(mo.FIG / "fig_transfer.pdf")
    mo.plt.close(fig)
    for o in OBJ:
        x = t[(t.objective == o) & (t.source != t.target)]
        k = mo.cname(o)
        mo.macro(f"Transfer{k}Mean", mo.pct(x.regret.mean()), "{:.1f}")
        mo.macro(f"Transfer{k}Max", mo.pct(x.regret.max()), "{:.1f}")
        mo.macro(f"Transfer{k}ShareUnderFive", mo.pct((x.regret < 0.05).mean()), "{:.0f}")
        mo.macro(f"Transfer{k}Median", mo.pct(x.regret.median()), "{:.1f}")
        mo.macro(f"Transfer{k}NOverFifty", int((x.regret > 0.5).sum()), "{:.0f}")
        mo.macro(f"Transfer{k}NPairs", len(x), "{:.0f}")
        w = x.loc[x.regret.idxmax()]
        a_, l_ = w.target.split("|")
        mo.macro(f"Transfer{k}WorstTarget", f"{mo.ACU_LAB[a_].lower()}, {l_}-load, {mo.CFG_LAB[w.config].lower()}", "{}")
        mo.macro(f"Transfer{k}NOverFiftyWorstTarget", int(((x.regret > 0.5) & (x.target == w.target) & (x.config == w.config)).sum()), "{:.0f}")
        for c in mo.CFG:
            mo.macro(f"Transfer{k}Median{mo.cname(c)}", mo.pct(x[x.config == c].regret.median()), "{:.1f}")
        # same acuity mix, other load vs other acuity mix
        sa = x[x.source.str.split("|").str[0] == x.target.str.split("|").str[0]]
        da = x[x.source.str.split("|").str[0] != x.target.str.split("|").str[0]]
        mo.macro(f"Transfer{k}SameAcuity", mo.pct(sa.regret.mean()), "{:.1f}")
        mo.macro(f"Transfer{k}OtherAcuity", mo.pct(da.regret.mean()), "{:.1f}")
        # best single donor per configuration
        donors = []
        for c in mo.CFG:
            y = t[(t.objective == o) & (t.config == c)]
            s = y.groupby("source").regret.agg(["mean", "max"]).sort_values("mean")
            donors.append((c, s.index[0], s.iloc[0]["mean"], s.iloc[0]["max"]))
        mo.macro(f"Transfer{k}DonorMean", mo.pct(np.mean([d_[2] for d_ in donors])), "{:.1f}")
        mo.macro(f"Transfer{k}DonorMax", mo.pct(max(d_[3] for d_ in donors)), "{:.1f}")
    return t


# ------------------------------------------------------------------ E8 robustness
def robustness_outputs(e2, opt_t):
    d = pd.read_csv(mo.RESULTS / "E8_robustness/robustness_kpis.csv.gz")
    d["routing"] = d["routing"].fillna("")
    reps = sorted(d.rep.unique())
    mo.macro("NRobReps", len(reps), "{:.0f}")
    base = e2[(e2.load == "medium") & e2.rep.isin(reps)]
    dflt_base = default_rows(base)
    pc = _P().policies_config()["default_policy"]
    vlab = {"phys_minus10": "Physician times $-10\\%$", "phys_plus10": "Physician times $+10\\%$",
            "diag_minus25": "Diagnostic times $-25\\%$", "diag_plus25": "Diagnostic times $+25\\%$",
            "cv_half": "All CVs $\\times0.5$", "cv_plus50": "All CVs $\\times1.5$",
            "profile_flat": "Flat arrival profile", "profile_peaked": "Peaked arrival profile",
            "pred_correlated": "Correlated prediction errors"}
    rows = []
    for v, dv in d.groupby("variant"):
        dv = dv.assign(load="medium")
        ddef = dv[(dv.queue == pc["queue"]) & ((dv.routing == pc["routing"]) | (dv.config == "POOLED"))]
        for (a, c), g in dv.groupby(["acuity", "config"]):
            gb = base[(base.acuity == a) & (base.config == c)]
            fixed_v = g[~g.queue.str.startswith("OPT")]
            res = {"variant": v, "acuity": a, "config": c}
            for k in ("ratp", "los_all_mean"):
                mv = fixed_v.groupby(["queue", "routing"])[k].mean()
                mb = gb.groupby(["queue", "routing"])[k].mean().reindex(mv.index)
                res[f"tau_{k}"] = stats.kendalltau(mv.to_numpy(), mb.to_numpy()).statistic
                best_b = mb.idxmin()
                res[f"same_best_{k}"] = mv.idxmin() == best_b
                res[f"regret_corebest_{k}"] = mv.loc[best_b] / mv.min() - 1
            # balanced objective: optimized policy vs best fixed under the perturbation
            gv = g.copy()
            gv["J"] = objective_frame(gv, ddef, "balanced")
            mj = gv.groupby(["queue", "routing"]).J.mean()
            fixed_j = mj[[not q.startswith("OPT") for q, _ in mj.index]]
            opt_j = mj[[q == "OPT-BALANCED" for q, _ in mj.index]]
            res["opt_bal_vs_best_fixed"] = (opt_j.iloc[0] / fixed_j.min() - 1) if len(opt_j) else np.nan
            res["opt_bal_gain_default"] = (1 - opt_j.iloc[0]) if len(opt_j) else np.nan
            rows.append(res)
    t = pd.DataFrame(rows)
    t.to_csv(mo.SUPP / "E8_robustness_summary.csv", index=False)
    order = list(vlab)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Kendall $\tau$ of policy ranking} & \multicolumn{2}{c}{Regret of core best (\%)} & \multicolumn{2}{c}{Optimized (balanced)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Perturbation & RATP & LOS & RATP & LOS & vs best fixed (\%) & vs default (\%) \\", r"\midrule"]
    for v in order:
        x = t[t.variant == v]
        if not len(x):
            continue
        lines.append(f"{vlab[v]} & {x.tau_ratp.mean():.2f} & {x.tau_los_all_mean.mean():.2f} & "
                     f"{mo.pct(x.regret_corebest_ratp.mean()):.1f} & {mo.pct(x.regret_corebest_los_all_mean.mean()):.1f} & "
                     f"{mo.pct(x.opt_bal_vs_best_fixed.mean()):+.1f} & {mo.pct(x.opt_bal_gain_default.mean()):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_robustness", "\n".join(lines))
    mo.macro("RobTauRatpMin", t.groupby("variant").tau_ratp.mean().min()); mo.macro("RobTauRatpMean", t.tau_ratp.mean())
    mo.macro("RobTauLosMin", t.groupby("variant").tau_los_all_mean.mean().min()); mo.macro("RobTauLosMean", t.tau_los_all_mean.mean())
    mo.macro("RobRegretRatpMean", mo.pct(t.regret_corebest_ratp.mean()), "{:.1f}")
    mo.macro("RobRegretRatpMax", mo.pct(t.regret_corebest_ratp.max()), "{:.1f}")
    mo.macro("RobRegretLosMean", mo.pct(t.regret_corebest_los_all_mean.mean()), "{:.1f}")
    mo.macro("RobRegretLosMax", mo.pct(t.regret_corebest_los_all_mean.max()), "{:.1f}")
    mo.macro("RobSameBestRatp", mo.pct(t.same_best_ratp.mean()), "{:.0f}")
    mo.macro("RobOptVsBestMean", mo.pct(t.opt_bal_vs_best_fixed.mean()), "{:.1f}")
    mo.macro("RobOptVsBestMax", mo.pct(t.opt_bal_vs_best_fixed.max()), "{:.1f}")
    mo.macro("RobOptGainDefaultMean", mo.pct(t.opt_bal_gain_default.mean()), "{:.1f}")
    mo.macro("RobWorstVariantRatp", vlab[t.groupby("variant").regret_corebest_ratp.mean().idxmax()], "{}")
    mo.macro("NRobVariants", t.variant.nunique(), "{:.0f}")
    return t


# ------------------------------------------------------------------ E9 out-of-sample
def oos_outputs():
    d = pd.read_csv(mo.RESULTS / "E9_out_of_sample/oos_kpis.csv.gz")
    d["routing"] = d["routing"].fillna("")
    guide = pd.read_csv(mo.SUPP / "decision_guide.csv")
    pc = _P().policies_config()["default_policy"]
    mo.macro("NOosTypes", d.id.nunique(), "{:.0f}")
    mo.macro("NOosReps", d.rep.nunique(), "{:.0f}")
    mo.macro("OosVisitsMin", d.visits_per_day.min(), "{:.0f}"); mo.macro("OosVisitsMax", d.visits_per_day.max(), "{:.0f}")
    rows = []
    for i, g in d.groupby("id"):
        cfg = g.config.iloc[0]
        dd = g[(g.queue == pc["queue"]) & ((g.routing == pc["routing"]) | (cfg == "POOLED"))].set_index("rep")
        g = g.copy()
        g["J"] = 0.5 * g.ratp / g.rep.map(dd.ratp) + 0.5 * g.los_all_mean / g.rep.map(dd.los_all_mean)
        fixed = g[~g.queue.str.startswith("OPT")]
        acu = g.near_acuity.iloc[0]
        grow = guide[(guide.acuity == acu) & (guide.config == cfg)].iloc[0]
        res = {"id": i, "config": cfg, "acuity_t": g.acuity_t.iloc[0], "load_value": g.load_value.iloc[0]}
        for k, col in (("ratp", "ratp_policy"), ("los_all_mean", "los_all_mean_policy"), ("J", "balanced_policy")):
            m = fixed.groupby(["queue", "routing"])[k].mean()
            best = m.min()
            pol = grow[col]
            q, r = (pol.split("/") + [""])[:2]
            rmap = {"C": "STREAM-CTAS", "P": "STREAM-PRED", "S": "STREAM-SIM", "": ""}
            key = (q, rmap[r] if cfg != "POOLED" else "")
            res[f"guide_{k}"] = m.loc[key] / best - 1
            res[f"default_{k}"] = m.loc[(pc["queue"], pc["routing"] if cfg != "POOLED" else "")] / best - 1
            opt = g[g.queue == ("OPT-ACCESS" if k == "ratp" else "OPT-BALANCED")].groupby("queue")[k].mean()
            res[f"opt_{k}"] = (opt.iloc[0] / best - 1) if len(opt) else np.nan
        rows.append(res)
    t = pd.DataFrame(rows)
    t.to_csv(mo.SUPP / "E9_out_of_sample_regret.csv", index=False)
    for k, nm in (("ratp", "Ratp"), ("los_all_mean", "Los"), ("J", "Bal")):
        for who in ("guide", "default", "opt"):
            mo.macro(f"Oos{who.title()}{nm}Mean", mo.pct(t[f"{who}_{k}"].mean()), "{:.1f}")
            mo.macro(f"Oos{who.title()}{nm}Max", mo.pct(t[f"{who}_{k}"].max()), "{:.1f}")
            mo.macro(f"Oos{who.title()}{nm}NGtTen", int((t[f"{who}_{k}"] > 0.10).sum()), "{:.0f}")
    fig, axes = mo.plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)
    for ax, (k, title) in zip(axes, (("ratp", "RATP"), ("los_all_mean", "Mean LOS"), ("J", "Balanced objective"))):
        data = [mo.pct(t[f"default_{k}"]), mo.pct(t[f"guide_{k}"]), mo.pct(t[f"opt_{k}"])]
        for j, (x, col) in enumerate(zip(data, [mo.PALETTE[1], mo.PALETTE[2], mo.PALETTE[0]])):
            jit = (np.random.default_rng(j).random(len(x)) - 0.5) * 0.25
            ax.scatter(np.full(len(x), j) + jit, x, s=10, color=col, alpha=0.85)
            ax.plot([j - 0.25, j + 0.25], [np.nanmedian(x)] * 2, color=mo.INK, lw=1.4)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Default", "Guide", "Optimized\n(nearest type)"], fontsize=6.5)
        ax.set_title(title, fontsize=8, loc="left", color=mo.INK)
        ax.axhline(0, color=mo.MUTED, lw=0.6)
    axes[0].set_ylabel("Regret vs best fixed policy (%)")
    fig.savefig(mo.FIG / "fig_oos.pdf")
    mo.plt.close(fig)
    return t


# ------------------------------------------------------------------ bootstrap and alternative acuity weights (E2)
def bootstrap_outputs(e2, B=1000, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for (a, l, c), g in e2.groupby(TYPE):
        for k in ("ratp", "los_all_mean"):
            M = g.pivot_table(index="rep", columns=["queue", "routing"], values=k)
            best = M.mean().idxmin()
            n = len(M)
            idx = rng.integers(0, n, size=(B, n))
            arr = M.to_numpy()
            means = arr[idx].mean(axis=1)
            win = np.argmin(means, axis=1)
            p = float(np.mean([M.columns[w] == best for w in win]))
            rows.append({"acuity": a, "load": l, "config": c, "kpi": k, "p_best": p})
    t = pd.DataFrame(rows)
    t.to_csv(mo.SUPP / "E2_bootstrap_best.csv", index=False)
    for k, nm in (("ratp", "Ratp"), ("los_all_mean", "Los")):
        x = t[t.kpi == k]
        mo.macro(f"Boot{nm}MedianP", mo.pct(x.p_best.median()), "{:.0f}")
        mo.macro(f"Boot{nm}NNinety", int((x.p_best >= 0.9).sum()), "{:.0f}")
        mo.macro(f"Boot{nm}NHalf", int((x.p_best < 0.5).sum()), "{:.0f}")
    mo.macro("BootB", B, "{:,.0f}")
    return t


def alt_weights_outputs(e2):
    """Best policy and best single policy when RATP uses other acuity weights."""
    schemes = {"inverse": [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5], "linear": [5, 4, 3, 2, 1], "equal": [1, 1, 1, 1, 1],
               "steep": [1, 1 / 4, 1 / 9, 1 / 16, 1 / 25]}
    m = e2.groupby(TYPE + ["queue", "routing"])[[f"ptp_ctas{l}" for l in range(1, 6)]].mean().reset_index()
    rows, bests = [], {}
    for nm, w in schemes.items():
        w = np.array(w, float) / np.sum(w) * np.sum(schemes["inverse"])
        m[nm] = sum(w[l - 1] * m[f"ptp_ctas{l}"].fillna(0) for l in range(1, 6))
        b = m.loc[m.groupby(TYPE)[nm].idxmin()].set_index(TYPE)
        bests[nm] = b[["queue", "routing"]].apply(tuple, axis=1)
        # best single policy (min mean relative regret)
        best_v = m.groupby(TYPE)[nm].transform("min")
        reg = m.assign(reg=m[nm] / best_v - 1)
        cands = []
        for q in mo.QUEUES:
            for r in mo.ROUTES:
                sel = (reg.queue == q) & ((reg.routing == r) | (reg.config == "POOLED"))
                cands.append(((q, r), reg[sel].reg.mean(), reg[sel].reg.max()))
        cands.sort(key=lambda x: x[1])
        rows.append({"scheme": nm, "osfa": mo.plabel(*cands[0][0]), "osfa_mean": cands[0][1], "osfa_max": cands[0][2]})
    t = pd.DataFrame(rows)
    for nm in schemes:
        t.loc[t.scheme == nm, "same_best_as_inverse"] = int((bests[nm] == bests["inverse"]).sum())
    t.to_csv(mo.SUPP / "E2_alternative_acuity_weights.csv", index=False)
    lab = {"inverse": r"$1/l$ (RATP)", "linear": r"$6-l$", "equal": "equal", "steep": r"$1/l^2$"}
    lines = [r"\begin{tabular}{lrlrr}", r"\toprule",
             r"Acuity weights & Same best as RATP & Best single policy & Mean regret (\%) & Max regret (\%) \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{lab[r.scheme]} & {int(r.same_best_as_inverse)}/{len(bests[r.scheme])} & {r.osfa} & {mo.pct(r.osfa_mean):.1f} & {mo.pct(r.osfa_max):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_alt_weights", "\n".join(lines))
    for nm in schemes:
        mo.macro(f"AltW{mo.cname(nm)}Same", int(t[t.scheme == nm].same_best_as_inverse.iloc[0]), "{:.0f}")
        mo.macro(f"AltW{mo.cname(nm)}Osfa", t[t.scheme == nm].osfa.iloc[0], "{}")
    return t


# ------------------------------------------------------------------ E6b levers
def lever_outputs():
    f = mo.RESULTS / "E6b_levers/lever_kpis.csv.gz"
    if not f.exists():
        print("WARNING: E6b missing; lever outputs skipped")
        return None
    d = pd.read_csv(f)
    d["objective"] = d["objective"].fillna("")
    dflt = d[d.variant == "DEFAULT"].set_index(TYPE + ["rep"])
    rows = []
    for o in OBJ:
        x = d[d.objective == o].copy()
        if o == "access":
            x["J"] = x.ratp
        else:
            r = dflt.reindex(x.set_index(TYPE + ["rep"]).index)
            x["J"] = 0.5 * x.ratp.to_numpy() / r.ratp.to_numpy() + 0.5 * x.los_all_mean.to_numpy() / r.los_all_mean.to_numpy()
        m = x.groupby(TYPE + ["variant"]).J.mean().unstack("variant")
        for v in [c for c in m.columns if c != "opt"]:
            ch = (m[v] / m["opt"] - 1).dropna()
            rows.append({"objective": o, "variant": v, "mean": ch.mean(), "min": ch.min(), "max": ch.max(),
                         "n_better": int((ch < -0.01).sum()), "n_worse": int((ch > 0.01).sum()), "n": len(ch)})
    t = pd.DataFrame(rows)
    t.to_csv(mo.SUPP / "E6b_lever_check.csv", index=False)
    lab = {"beta=0": r"No reassessment bonus ($\beta=0$)", "offset=0": r"No routing offset ($b=0$)"}
    order = [f"gamma={g}" for g in (0.02, 0.05, 0.1, 0.2, 0.4)] + ["beta=0", "offset=0"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Access objective} & \multicolumn{3}{c}{Balanced objective} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Change to the optimized rule & Mean change (\%) & Better & Worse & Mean change (\%) & Better & Worse \\", r"\midrule"]
    for v in order:
        cells = []
        for o in OBJ:
            y = t[(t.objective == o) & (t.variant == v)]
            cells += [f"{mo.pct(y['mean'].iloc[0]):+.1f}", f"{y.n_better.iloc[0]}/{y.n.iloc[0]}", f"{y.n_worse.iloc[0]}/{y.n.iloc[0]}"] if len(y) else ["", "", ""]
        name = lab.get(v, r"Ageing $\gamma=" + v.split("=")[1] + "$")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mo.write_tex("tab_levers", "\n".join(lines))
    for o in OBJ:
        k = mo.cname(o)
        y = t[t.objective == o].set_index("variant")
        g = y.loc[[v for v in y.index if v.startswith("gamma")]]
        mo.macro(f"Lever{k}GammaBestMean", mo.pct(g["mean"].min()), "{:+.1f}")
        mo.macro(f"Lever{k}GammaNBetter", int(g.n_better.max()), "{:.0f}")
        mo.macro(f"Lever{k}GammaLargeMean", mo.pct(y.loc[max(g.index, key=lambda v: float(v.split("=")[1])), "mean"]), "{:+.1f}")
        mo.macro(f"Lever{k}BetaZeroMean", mo.pct(y.loc["beta=0", "mean"]), "{:+.1f}")
        mo.macro(f"Lever{k}BetaZeroMax", mo.pct(y.loc["beta=0", "max"]), "{:+.1f}")
        mo.macro(f"Lever{k}BetaZeroNWorse", int(y.loc["beta=0", "n_worse"]), "{:.0f}")
        mo.macro(f"Lever{k}OffsetZeroMean", mo.pct(y.loc["offset=0", "mean"]), "{:+.1f}")
        mo.macro(f"Lever{k}OffsetZeroMax", mo.pct(y.loc["offset=0", "max"]), "{:+.1f}")
        mo.macro(f"Lever{k}OffsetZeroNWorse", int(y.loc["offset=0", "n_worse"]), "{:.0f}")
    mo.macro("NLeverReps", d.rep.nunique(), "{:.0f}")
    gs = sorted(float(v.split("=")[1]) for v in d.variant.unique() if v.startswith("gamma"))
    mo.macro("LeverGammaMin", gs[0], "{:g}"); mo.macro("LeverGammaMax", gs[-1], "{:g}")
    return t


def build_all(e2):
    t, specs = optimization_outputs(e2)
    lever_outputs()
    transfer_outputs()
    robustness_outputs(e2, t)
    oos_outputs()
    bootstrap_outputs(e2)
    alt_weights_outputs(e2)
