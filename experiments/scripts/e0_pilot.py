"""E0 pilot: warm-up length (MSER-5) and number of replications (CI half-width rule).

Warm-up: runs with no warm-up deletion; series = mean LOS of patients by arrival hour,
averaged over pilot replications; MSER-5 truncation point searched over the first half.
Replications: n0 pilot replications of every core ED type under the default policy;
n needed so that the 95% CI half-width is at most 5% of the mean for mean LOS and RATP.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from common import P, pmap, save
from edsim.generator import generate
from edsim.kpis import event_log_frame
from edsim.runner import kpi_row, run_policy

N0 = 10
REL = 0.05
DEFAULT = P.policies_config()["default_policy"]


def mser5(series: np.ndarray) -> int:
    """Truncation point (in original series units) minimising the MSER statistic over batches of 5."""
    m = len(series) // 5
    b = series[: m * 5].reshape(m, 5).mean(axis=1)
    best, best_d = math.inf, 0
    for d in range(0, m // 2):
        x = b[d:]
        stat = x.var(ddof=0) / len(x)
        if stat < best:
            best, best_d = stat, d
    return best_d * 5


def _warm_task(args):
    acuity, config, rep = args
    rc = P.parameters()["run_control"]
    rc["warmup_days"]["value"] = 0
    ed = P.build_ed_type(acuity, "high", config)
    pts = generate(ed, 100 + rep)
    rt = DEFAULT["routing"] if config != "POOLED" else None
    eng = run_policy(ed, pts, DEFAULT["queue"], rt)
    log = event_log_frame(eng)
    dep = log.groupby("pid").t_end.max()
    arr = pd.Series({p.pid: p.arrival for p in pts})
    obs = arr[arr < eng.obs_end]
    los = (dep.reindex(obs.index) - obs)
    hour = (obs // 60).astype(int)
    s = los.groupby(hour).mean()
    return acuity, config, rep, s.reindex(range(int(eng.obs_end // 60))).to_numpy()


def _rep_task(args):
    acuity, load, config, rep = args
    ed = P.build_ed_type(acuity, load, config)
    pts = generate(ed, 100 + rep)
    rt = DEFAULT["routing"] if config != "POOLED" else None
    eng = run_policy(ed, pts, DEFAULT["queue"], rt)
    return kpi_row(ed, 100 + rep, DEFAULT["queue"], rt, 1.0, eng)


def main():
    f = P.design()["factors"]
    # ---- warm-up
    tasks = [(a, c, r) for a in f["acuity"] for c in f["config"] for r in range(N0)]
    res = pmap(_warm_task, tasks)
    rows = []
    for a in f["acuity"]:
        for c in f["config"]:
            mats = np.vstack([s for (aa, cc, _, s) in res if aa == a and cc == c])
            series = np.nanmean(mats, axis=0)
            d = mser5(series)
            rows.append({"acuity": a, "config": c, "load": "high", "mser5_truncation_hours": d,
                         "series_hours": len(series)})
    warm = pd.DataFrame(rows)
    save(warm, "warmup_mser5", "E0_pilot", pilot_replications=N0, policy=DEFAULT)
    print(warm)
    print("max truncation (h):", warm.mser5_truncation_hours.max())

    # ---- replications
    tasks = [(a, l, c, r) for a in f["acuity"] for l in f["load"] for c in f["config"] for r in range(N0)]
    kp = pd.DataFrame(pmap(_rep_task, tasks))
    save(kp, "pilot_kpis", "E0_pilot", pilot_replications=N0, policy=DEFAULT)
    t = stats.t.ppf(0.975, N0 - 1)
    out = []
    for key, g in kp.groupby(["acuity", "load", "config"]):
        r = dict(zip(["acuity", "load", "config"], key))
        for k in ("los_all_mean", "ratp", "pia_all_p90"):
            m, s = g[k].mean(), g[k].std(ddof=1)
            rh = t * s / math.sqrt(N0) / m
            r[f"{k}_relhw"] = rh
            r[f"{k}_n_needed"] = math.ceil(N0 * (rh / REL) ** 2)
        out.append(r)
    need = pd.DataFrame(out)
    save(need, "replications_needed", "E0_pilot", pilot_replications=N0, rel_halfwidth=REL)
    print(need.to_string())


if __name__ == "__main__":
    main()
