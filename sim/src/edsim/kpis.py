"""KPIs computed from the patient-level event log only.

One definition per KPI (see MODEL_SPEC.md Section 7). Aggregate ("all patients")
values are computed from patient records, never from other KPI values.
Time stamps follow the ED Benchmarking Alliance definitions (ED-Metrics_WELCH2013):
door = arrival; time to physician = door to start of the physician initial
assessment (PIA); LOS = door to departure; boarding = decision to departure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P

LOG_COLUMNS = ["pid", "stage", "resource", "t_req", "t_start", "t_end"]


def event_log_frame(engine) -> pd.DataFrame:
    return pd.DataFrame(engine.log, columns=LOG_COLUMNS)


def patient_table(engine, log: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per observed patient (arrival inside the observation window)."""
    if log is None:
        log = event_log_frame(engine)
    pts = engine.pts
    base = pd.DataFrame({
        "pid": [p.pid for p in pts], "arrival": [p.arrival for p in pts],
        "ctas": [p.ctas for p in pts], "admit": [p.admit for p in pts],
        "pathway": [p.pathway for p in pts], "routed_second": engine.routed_second,
    })
    base = base[(base.arrival >= engine.warm) & (base.arrival < engine.obs_end)].set_index("pid")
    g = log[log.pid.isin(base.index)]
    first = g.groupby(["pid", "stage"]).t_start.min().unstack()
    base["t_triage"] = first.get("TRIAGE")
    base["t_space"] = g[g.stage == "SPACE"].groupby("pid").t_start.min()
    base["t_pia"] = first.get("AS")
    base["t_decision"] = first.get("DECISION")
    base["t_depart"] = g.groupby("pid").t_end.max()
    ends = g[g.stage.isin(["DD", "BOARD"])].groupby("pid").t_end.max()
    base["finished"] = base.index.isin(ends.index)
    return base


# ---------------------------------------------------------------- KPI definitions
def door_to_physician(pt):           # minutes
    return pt.t_pia - pt.arrival


def time_to_space(pt):
    return pt.t_space - pt.arrival


def door_to_triage(pt):
    return pt.t_triage - pt.arrival


def length_of_stay(pt):
    return pt.t_depart - pt.arrival


def boarding_time(pt):
    return (pt.t_depart - pt.t_decision).where(pt.admit)


def ptp_by_ctas(pt) -> dict:
    """Share of patients of each CTAS level whose door-to-physician time exceeds the target."""
    ub = P.v(P.parameters()["ctas_targets_minutes"])
    w = door_to_physician(pt)
    out = {}
    for l in range(1, 6):
        sel = pt.ctas == l
        out[l] = float((w[sel] > ub[l - 1]).mean()) if sel.any() else np.nan
    return out


def ratp(pt) -> float:
    """RATP = sum_l (1/l) PTP_l (acuity-weighted breach rate)."""
    return float(sum(v / l for l, v in ptp_by_ctas(pt).items() if not np.isnan(v)))


def awt(pt) -> float:
    """AWT = sum_l (1/l) mean_l max(0, door-to-physician - target_l), minutes (acuity-weighted lateness)."""
    ub = P.v(P.parameters()["ctas_targets_minutes"])
    w = door_to_physician(pt)
    tot = 0.0
    for l in range(1, 6):
        sel = pt.ctas == l
        if sel.any():
            tot += float(np.maximum(w[sel] - ub[l - 1], 0.0).mean()) / l
    return tot


def _stats(x: pd.Series, prefix: str, out: dict):
    x = x.dropna()
    out[prefix + "_mean"] = float(x.mean()) if len(x) else np.nan
    out[prefix + "_p50"] = float(x.quantile(0.5)) if len(x) else np.nan
    out[prefix + "_p90"] = float(x.quantile(0.9)) if len(x) else np.nan
    out[prefix + "_p95"] = float(x.quantile(0.95)) if len(x) else np.nan
    out[prefix + "_max"] = float(x.max()) if len(x) else np.nan


def utilisation(engine, log: pd.DataFrame) -> dict:
    t0, t1 = engine.warm, engine.obs_end
    busy = log[log.stage.isin(["SPACE", "AS", "RA", "LB", "XR", "CT", "DR"])].copy()
    busy["dur"] = (busy.t_end.clip(t0, t1) - busy.t_start.clip(t0, t1)).clip(lower=0)
    cap = engine.capacity_minutes(t0, t1)
    out = {}
    for res, minutes in cap.items():
        if minutes <= 0:
            continue
        sel = busy.resource == res
        out["util_" + res] = float(busy.loc[sel, "dur"].sum() / minutes)
    return out


def compute_kpis(engine) -> dict:
    log = event_log_frame(engine)
    pt = patient_table(engine, log)
    out = {"n_patients": int(len(pt)), "n_unfinished": int((~pt.finished).sum())}
    days = (engine.obs_end - engine.warm) / P.MIN_PER_DAY
    out["throughput_per_day"] = len(pt) / days
    out["share_second_area"] = float(pt.routed_second.mean())
    out["admit_share"] = float(pt.admit.mean())
    _stats(door_to_triage(pt), "d2t_all", out)
    _stats(door_to_physician(pt), "pia_all", out)
    _stats(time_to_space(pt), "ttb_all", out)
    _stats(length_of_stay(pt), "los_all", out)
    _stats(length_of_stay(pt)[~pt.admit], "los_disch", out)
    _stats(length_of_stay(pt)[pt.admit], "los_adm", out)
    _stats(boarding_time(pt), "board", out)
    for l in range(1, 6):
        sel = pt.ctas == l
        _stats(door_to_physician(pt)[sel], f"pia_ctas{l}", out)
        _stats(time_to_space(pt)[sel], f"ttb_ctas{l}", out)
        _stats(length_of_stay(pt)[sel], f"los_ctas{l}", out)
    for l, val in ptp_by_ctas(pt).items():
        out[f"ptp_ctas{l}"] = val
    out["ptp_all"] = float((door_to_physician(pt) > pt.ctas.map(
        dict(enumerate(P.v(P.parameters()["ctas_targets_minutes"]), start=1)))).mean())
    out["ratp"] = ratp(pt)
    out["awt"] = awt(pt)
    # stage waiting times (request to start), observed patients
    g = log[log.pid.isin(pt.index)]
    for st in ["TRIAGE", "SPACE", "AS", "RA", "LB", "XR", "CT", "DR"]:
        s = g[g.stage == st]
        out[f"wait_{st}_mean"] = float((s.t_start - s.t_req).mean()) if len(s) else np.nan
    out.update(utilisation(engine, log))
    return out
