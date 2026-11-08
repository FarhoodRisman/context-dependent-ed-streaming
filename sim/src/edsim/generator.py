"""Synthetic patient generator.

One call creates the complete patient list for one ED type and replication:
arrival times, CTAS, disposition, pathway, every activity duration and the
predicted attributes at every prediction-error level alpha. Every policy is
then run on the same list (common random numbers by construction).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import params as P

# code attribute order: A CTAS, B service class, C admission, D diagnostics, E resource use, F consult
CODE_RANGES = (4, 4, 1, 1, 3, 1)
N_CLASSES = (5, 5, 2, 2, 4, 2)


@dataclass(frozen=True)
class Patient:
    pid: int
    arrival: float
    ctas: int
    admit: bool
    pathway: str
    steps: tuple
    d_triage: float
    d_reg: float
    d_as: float
    d_ra: float
    d_steps: tuple
    d_end: float           # discharge process (discharged) or boarding (admitted)
    service_total: float   # total treatment time (no waits, no boarding)
    code_true: tuple
    pred: dict             # alpha -> predicted code (A is always the true CTAS)


def seed_for(*parts) -> np.random.SeedSequence:
    base = P.v(P.parameters()["run_control"]["base_seed"])
    return np.random.SeedSequence([base, *[int(p) for p in parts]])


# ---------------------------------------------------------------- confusion matrices
def reference_confusion(n_classes: int, accuracy: float, decay: float) -> np.ndarray:
    m = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        w = np.array([0.0 if j == i else decay ** (abs(i - j) - 1) for j in range(n_classes)])
        m[i] = (1 - accuracy) * w / w.sum()
        m[i, i] = accuracy
    return m


def confusion_at(alpha: float, ref: np.ndarray) -> np.ndarray:
    m = np.eye(ref.shape[0]) + alpha * (ref - np.eye(ref.shape[0]))
    if (m < -1e-12).any():
        raise ValueError(f"alpha={alpha} gives negative probabilities")
    return np.clip(m, 0.0, None)


@lru_cache(maxsize=None)
def confusion_matrices() -> dict:
    pr = P.parameters()["prediction"]
    acc = {k: P.v(x) for k, x in pr["reference_accuracy"].items()}
    decay = P.v(pr["ordinal_error_decay"])
    names = ("service", "admission", "diagnostics", "resource", "consult")
    sizes = (5, 2, 2, 4, 2)
    return {n: reference_confusion(k, acc[n], decay) for n, k in zip(names, sizes)}


# ---------------------------------------------------------------- service-class cut points
@lru_cache(maxsize=None)
def service_cutpoints() -> tuple:
    """Cut points of total treatment time for the five service classes, from a
    reference sample of the community ED with typical diagnostics (fixed seed)."""
    rng = np.random.default_rng(seed_for(999, 0))
    sample = _sample_attributes(rng, 200_000, "community", "typical", "low")["service_total"]  # reference ED, unperturbed
    q = P.v(P.parameters()["prediction"]["service_class_quantiles"])
    return tuple(float(x) for x in np.quantile(sample, q))


def service_class(total: np.ndarray) -> np.ndarray:
    """1 = very long ... 5 = very short."""
    cuts = service_cutpoints()
    longness = np.searchsorted(cuts, total, side="right")  # 0 short .. 4 long
    return 5 - longness


# ---------------------------------------------------------------- sampling
def _lognormal(rng, mean, cv, size):
    mu, s = P.lognormal_params(mean, cv)
    return rng.lognormal(mu, s, size)


def _sample_attributes(rng, n, acuity, diag_level, boarding, perturb=None):
    """acuity: name or CTAS-share vector; diag_level: name or shift; boarding: name or mean.
    perturb (robustness only): phys_mult, diag_mult multiply mean durations; cv_mult multiplies CVs."""
    par = P.parameters()
    d = par["durations"]
    pt = perturb or {}
    pm, dm, cm = pt.get("phys_mult", 1.0), pt.get("diag_mult", 1.0), pt.get("cv_mult", 1.0)
    mix = np.array(P.mix_value(acuity))
    ctas = rng.choice(np.arange(1, 6), size=n, p=mix / mix.sum())
    p_adm = np.array(P.v(par["admission_probability_by_ctas"]))
    admit = rng.random(n) < p_adm[ctas - 1]
    pws = P.pathways()
    path_idx = np.empty(n, dtype=int)
    u = rng.random(n)
    for c in range(1, 6):
        for a in (True, False):
            sel = (ctas == c) & (admit == a)
            if sel.any():
                cdf = np.cumsum(P.pathway_probs(c, a, diag_level))
                path_idx[sel] = np.minimum(np.searchsorted(cdf, u[sel], side="right"), len(cdf) - 1)
    as_mean = np.array(d["physician_initial_by_ctas"]["mean"], dtype=float)[ctas - 1] * pm
    mu_s2 = np.log(1 + (d["physician_initial_by_ctas"]["cv"] * cm) ** 2)
    d_as = rng.lognormal(np.log(as_mean) - mu_s2 / 2, math.sqrt(mu_s2))
    d_ra = _lognormal(rng, d["physician_reassessment"]["mean"] * pm, d["physician_reassessment"]["cv"] * cm, n)
    step_d = {s: _lognormal(rng, d[s]["mean"] * dm, d[s]["cv"] * cm, n) for s in P.DIAG_STAGES}
    d_dd = _lognormal(rng, d["discharge_process"]["mean"], d["discharge_process"]["cv"] * cm, n)
    bcv = d["boarding"]["low"]["cv"] if not isinstance(boarding, str) else d["boarding"][boarding]["cv"]
    d_board = _lognormal(rng, P.boarding_mean_value(boarding), bcv * cm, n)
    d_tri = _lognormal(rng, d["triage"]["mean"], d["triage"]["cv"] * cm, n)
    d_reg = _lognormal(rng, d["registration"]["mean"], d["registration"]["cv"] * cm, n)
    steps = [pws[i].steps for i in path_idx]
    has_diag = np.array([len(s) > 0 for s in steps])
    diag_sum = np.array([sum(step_d[s][k] for s in st) for k, st in enumerate(steps)])
    service_total = d_as + np.where(has_diag, d_ra, 0.0) + diag_sum + np.where(admit, 0.0, d_dd)
    return dict(ctas=ctas, admit=admit, path_idx=path_idx, steps=steps, d_as=d_as, d_ra=d_ra,
                step_d=step_d, d_dd=d_dd, d_board=d_board, d_tri=d_tri, d_reg=d_reg,
                service_total=service_total)


def _arrivals(rng, rate_per_day, horizon_min, profile_power=1.0):
    prof = P.hourly_profile(profile_power)
    lam_max = rate_per_day / P.MIN_PER_DAY * prof.max()
    n_cand = rng.poisson(lam_max * horizon_min * 1.0) + 50
    t = np.cumsum(rng.exponential(1.0 / lam_max, n_cand))
    while t[-1] < horizon_min:     # extend if needed
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(1.0 / lam_max, n_cand))])
    t = t[t < horizon_min]
    hours = ((t // 60) % 24).astype(int)
    keep = rng.random(len(t)) < prof[hours] / prof.max()
    return t[keep]


def generate(ed, rep: int, alphas=None) -> list:
    """Patient list for one ED type and replication."""
    rc = P.parameters()["run_control"]
    horizon = (P.v(rc["warmup_days"]) + P.v(rc["observation_days"]) + P.v(rc["max_cooldown_days"])) * P.MIN_PER_DAY
    ss = seed_for(*ed.seed_parts, rep)
    s_arr, s_attr, s_pred = ss.spawn(3)
    pt = ed.perturb or {}
    arr = _arrivals(np.random.default_rng(s_arr), ed.visits_per_day, horizon, pt.get("profile_power", 1.0))
    n = len(arr)
    at = _sample_attributes(np.random.default_rng(s_attr), n, list(ed.mix), ed.diag_shift, ed.board_mean, pt)

    # true codes
    ctas = at["ctas"]
    B = service_class(at["service_total"])
    C = np.where(at["admit"], 1, 2)
    D = np.array([1 if any(s in ("LB", "XR", "CT") for s in st) else 2 for st in at["steps"]])
    nres = np.array([len(st) for st in at["steps"]])
    E = np.select([nres >= 3, nres == 2, nres == 1], [1, 2, 3], default=4)
    F = np.array([1 if "DR" in st else 2 for st in at["steps"]])
    true = np.stack([ctas, B, C, D, E, F], axis=1)

    # predicted codes: one uniform per patient and attribute, shared by all alphas
    if alphas is None:
        alphas = sorted(set(P.v(P.parameters()["prediction"]["alpha_sweep"])) | {P.v(P.parameters()["prediction"]["alpha_main"])})
    U = np.random.default_rng(s_pred).random((n, 5))
    if pt.get("corr_pred"):
        U[:, 1:] = U[:, :1]       # one draw for all attributes: prediction errors move together
    cms = confusion_matrices()
    attr_cols = (("service", 1, 5), ("admission", 2, 2), ("diagnostics", 3, 2), ("resource", 4, 4), ("consult", 5, 2))
    preds = {}
    for a in alphas:
        pc = true.copy()
        for j, (name, col, k) in enumerate(attr_cols):
            cdf = np.cumsum(confusion_at(a, cms[name]), axis=1)
            rows = cdf[true[:, col] - 1]
            pc[:, col] = np.minimum((rows < U[:, j:j + 1]).sum(axis=1), k - 1) + 1
        preds[a] = pc

    pws = P.pathways()
    out = []
    for i in range(n):
        st = at["steps"][i]
        out.append(Patient(
            pid=i, arrival=float(arr[i]), ctas=int(ctas[i]), admit=bool(at["admit"][i]),
            pathway=pws[at["path_idx"][i]].name, steps=st,
            d_triage=float(at["d_tri"][i]), d_reg=float(at["d_reg"][i]),
            d_as=float(at["d_as"][i]), d_ra=float(at["d_ra"][i]),
            d_steps=tuple(float(at["step_d"][s][i]) for s in st),
            d_end=float(at["d_board"][i] if at["admit"][i] else at["d_dd"][i]),
            service_total=float(at["service_total"][i]),
            code_true=tuple(int(x) for x in true[i]),
            pred={a: tuple(int(x) for x in preds[a][i]) for a in alphas},
        ))
    return out
