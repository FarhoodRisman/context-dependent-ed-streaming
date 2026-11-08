"""Automated checks that every results file must pass before outputs are built
(number-integrity rule 6). Exits with status 1 on any failure; writes checks.json."""
import json
import sys

import numpy as np
import pandas as pd

from common import RESULTS

FILES = {"E2": "E2_main/kpis.csv.gz", "E4": "E4_alpha/kpis_alpha.csv.gz",
         "E5": "E5_sensitivity/kpis_sensitivity.csv.gz", "E1": "E1_validation/face_validity_kpis.csv.gz"}
KEY = ["acuity", "load", "config", "boarding", "diagnostics", "rep", "queue", "routing", "alpha"]
COMPARE = ["los_all_mean", "ratp", "awt", "pia_all_p90", "n_patients"]


def main():
    out, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        out.append({"check": name, "pass": bool(cond), "detail": detail})

    data = {k: pd.read_csv(RESULTS / f) for k, f in FILES.items() if (RESULTS / f).exists()}
    for k, df in data.items():
        df["routing"] = df["routing"].fillna("")
        check(f"{k}: every observed patient departed", (df.n_unfinished == 0).all(),
              f"rows with unfinished patients: {(df.n_unfinished > 0).sum()}")
        ptp = df[[c for c in df if c.startswith("ptp_")]]
        check(f"{k}: breach rates in [0,1]", ((ptp >= 0) & (ptp <= 1) | ptp.isna()).all().all())
        util = df[[c for c in df if c.startswith("util_")]]
        check(f"{k}: utilisation in [0,1]", ((util >= 0) & (util <= 1 + 1e-9) | util.isna()).all().all())
        g = df[[f"los_ctas{l}_mean" for l in range(1, 6)]]
        within = (df.los_all_mean >= g.min(axis=1) - 1e-9) & (df.los_all_mean <= g.max(axis=1) + 1e-9)
        check(f"{k}: overall mean LOS within CTAS group means", within.all(), f"violations: {(~within).sum()}")
        within2 = (df.los_all_mean >= df[["los_disch_mean", "los_adm_mean"]].min(axis=1) - 1e-9) & \
                  (df.los_all_mean <= df[["los_disch_mean", "los_adm_mean"]].max(axis=1) + 1e-9)
        check(f"{k}: overall mean LOS within disposition group means", within2.all())
        p = df[[f"pia_ctas{l}_mean" for l in range(1, 6)]]
        within3 = (df.pia_all_mean >= p.min(axis=1) - 1e-9) & (df.pia_all_mean <= p.max(axis=1) + 1e-9)
        check(f"{k}: overall mean door-to-physician within CTAS group means", within3.all())
        check(f"{k}: no duplicate run keys", not df.duplicated(KEY).any())

    # same policy + same seed => identical numbers across experiments
    def same(a, b, label):
        if a not in data or b not in data:
            return
        m = data[a].merge(data[b], on=KEY, suffixes=("_a", "_b"))
        diff = max(float(np.nanmax(np.abs(m[f"{c}_a"] - m[f"{c}_b"]))) if len(m) else 0.0 for c in COMPARE)
        check(f"{label}: identical results for shared runs", len(m) > 0 and diff == 0.0,
              f"shared runs: {len(m)}, max abs difference: {diff}")
    same("E2", "E4", "E2 vs E4 (medium load, alpha = 1)")
    same("E2", "E1", "E2 vs E1 (community pooled)")
    f3 = RESULTS / "E3_fidelity/fidelity.csv.gz"
    if f3.exists() and "E2" in data:
        e3 = pd.read_csv(f3)
        e3 = e3[(e3.level == "queue") & e3.representation.str.startswith("SIMREP")]
        e2 = data["E2"]
        e2 = e2[e2.routing == "STREAM-CTAS"]
        m = e3.merge(e2, left_on=["acuity", "load", "config", "rep", "reference"],
                     right_on=["acuity", "load", "config", "rep", "queue"])
        d = float(np.abs(m.los_all_mean_ref - m.los_all_mean).max()) if len(m) else 0.0
        check("E3 reference rules vs E2: identical LOS for shared runs", len(m) > 0 and d == 0.0,
              f"shared runs: {len(m)}, max abs difference: {d}")
    (RESULTS / "checks.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(("PASS " if r["pass"] else "FAIL ") + r["check"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
