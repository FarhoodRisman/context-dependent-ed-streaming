"""Load the configuration files and derive the quantities that depend on them.

Every input number lives in sim/configs/*.json. This module only reads those
numbers and derives capacities, rates and probabilities from them; it types
no model inputs of its own.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "sim" / "configs"

DIAG_STAGES = ("LB", "XR", "CT", "DR")
MIN_PER_DAY = 1440.0


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def parameters() -> dict:
    return _load("parameters.json")


@lru_cache(maxsize=None)
def policies_config() -> dict:
    return _load("policies.json")


@lru_cache(maxsize=None)
def design() -> dict:
    return _load("design.json")


def v(entry):
    """Return the 'value' of a parameter entry."""
    return entry["value"] if isinstance(entry, dict) and "value" in entry else entry


# ---------------------------------------------------------------- pathways
@dataclass(frozen=True)
class Pathway:
    name: str
    steps: tuple          # diagnostic/consult steps in order, e.g. ("LB", "XR")
    admit: bool
    base_freq: float      # published relative frequency


@lru_cache(maxsize=None)
def pathways() -> tuple:
    path = REPO_ROOT / parameters()["pathways"]["file"]
    out = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stages = row["pathway"].split(">")
            steps = tuple(s for s in stages if s in DIAG_STAGES)
            out.append(Pathway(row["rank"], steps, row["ends_in"] == "admission", float(row["count"])))
    return tuple(out)


def diag_shift_value(diag_level) -> float:
    """Numeric diagnostic-intensity shift for a named level (or a number passed through)."""
    if isinstance(diag_level, str):
        return v(parameters()["pathways"]["diagnostic_intensity_shift"])[diag_level]
    return float(diag_level)


def pathway_probs(ctas: int, admit: bool, diag_level) -> np.ndarray:
    """P(pathway | CTAS, disposition) over all pathways (zero for the other disposition).
    diag_level is a named level ('light', 'typical', 'heavy') or a numeric shift."""
    p = parameters()["pathways"]
    tilt = v(p["diagnostic_tilt_by_ctas"])[ctas - 1] + diag_shift_value(diag_level)
    w = np.array([pw.base_freq * math.exp(tilt * len(pw.steps)) if pw.admit == admit else 0.0
                  for pw in pathways()])
    return w / w.sum()


# ---------------------------------------------------------------- durations
def lognormal_params(mean: float, cv: float) -> tuple:
    s2 = math.log(1.0 + cv * cv)
    return math.log(mean) - s2 / 2.0, math.sqrt(s2)


def boarding_mean_value(boarding) -> float:
    if isinstance(boarding, str):
        return parameters()["durations"]["boarding"][boarding]["mean"]
    return float(boarding)


def duration_means(boarding) -> dict:
    d = parameters()["durations"]
    return {
        "triage": d["triage"]["mean"], "registration": d["registration"]["mean"],
        "AS": d["physician_initial_by_ctas"]["mean"], "RA": d["physician_reassessment"]["mean"],
        "LB": d["LB"]["mean"], "XR": d["XR"]["mean"], "CT": d["CT"]["mean"], "DR": d["DR"]["mean"],
        "DD": d["discharge_process"]["mean"], "BOARD": boarding_mean_value(boarding),
    }


def mix_value(acuity) -> list:
    """CTAS shares for a named acuity mix (or a 5-vector passed through)."""
    if isinstance(acuity, str):
        return v(parameters()["acuity_mix"][acuity])
    return list(acuity)


# ---------------------------------------------------------------- expected workloads
def expected_workloads(acuity, diag_level, boarding, ctas_subset=None) -> dict:
    """Expected work per arriving patient (minutes), optionally only for some CTAS levels
    (the returned values are still per arriving patient, i.e. weighted by the CTAS share).
    acuity, diag_level and boarding are names or numeric values (mix vector, shift, mean)."""
    mix = mix_value(acuity)
    p_adm = v(parameters()["admission_probability_by_ctas"])
    m = duration_means(boarding)
    pws = pathways()
    out = {"physician": 0.0, "space": 0.0, "LB": 0.0, "XR": 0.0, "CT": 0.0, "DR": 0.0}
    for c in range(1, 6):
        if ctas_subset is not None and c not in ctas_subset:
            continue
        for admit, pa in ((True, p_adm[c - 1]), (False, 1 - p_adm[c - 1])):
            probs = pathway_probs(c, admit, diag_level)
            for pw, pp in zip(pws, probs):
                if pp == 0:
                    continue
                w = mix[c - 1] * pa * pp
                has_diag = len(pw.steps) > 0
                phys = m["AS"][c - 1] + (m["RA"] if has_diag else 0.0)
                diag = sum(m[s] for s in pw.steps)
                end = m["BOARD"] if admit else m["DD"]
                out["physician"] += w * phys
                out["space"] += w * (phys + diag + end)
                for s in pw.steps:
                    out[s] += w * m[s]
    return out


def hourly_profile(power: float = 1.0) -> np.ndarray:
    """Hourly arrival profile normalised to mean 1; power != 1 flattens (<1) or sharpens (>1) it."""
    prof = np.array(v(parameters()["arrivals"]["hourly_profile"]), dtype=float)
    if power != 1.0:
        prof = prof ** power
    return prof / prof.mean()


# ---------------------------------------------------------------- ED type
@dataclass
class EDType:
    acuity: str
    load: str
    config: str
    boarding: str = "low"
    diagnostics: str = "typical"
    # derived
    visits_per_day: float = 0.0
    load_multiplier: float = 1.0
    physicians_by_shift: list = field(default_factory=list)          # total per shift
    second_physicians_by_shift: list = field(default_factory=list)   # FASTTRACK only
    spaces_total: int = 0
    spaces_second: int = 0
    diag_capacity: dict = field(default_factory=dict)
    triage_nurses: int = 0
    offered_physician_load: float = 0.0
    # demand inputs (names resolved to values); capacities above stay at core sizing
    mix: tuple = ()
    diag_shift: float = 0.0
    board_mean: float = 0.0
    perturb: dict = field(default_factory=dict)   # phys_mult, diag_mult, cv_mult, profile_power, corr_pred
    seed_parts: tuple = ()

    @property
    def key(self) -> str:
        return f"{self.acuity}|{self.load}|{self.config}|{self.boarding}|{self.diagnostics}"

    @property
    def areas(self) -> list:
        return design()["configurations"][self.config]["areas"]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def build_ed_type(acuity, load, config: str, boarding="low", diagnostics="typical",
                  perturb: dict | None = None, seed_parts: tuple | None = None, label: str | None = None) -> EDType:
    """Derive capacities and arrival rate for one ED type.

    Capacities are sized for the 'medium' load level with the core boarding and
    diagnostic levels, so that the sensitivity factors change demand but not capacity.
    acuity may be a name or a CTAS-share vector; load a name or an offered-load value;
    boarding a name or a mean (minutes); diagnostics a name or a numeric shift.
    perturb changes demand-side inputs only (robustness experiments).
    """
    P = parameters()
    cs = P["capacity_sizing"]
    core = design()["core"]
    base_rate = v(P["arrivals"]["base_visits_per_day"])
    prof = hourly_profile()
    shifts = v(cs["shifts_start_hours"])
    rho_med = v(cs["physician_target_utilisation_medium"])

    wl = expected_workloads(acuity, core["diagnostics"], core["boarding"])
    # the two patient streams used for sizing: CTAS 1-3 (main) and CTAS 4-5 (second area)
    wl_hi = expected_workloads(acuity, core["diagnostics"], core["boarding"], ctas_subset=(1, 2, 3))
    wl_low = expected_workloads(acuity, core["diagnostics"], core["boarding"], ctas_subset=(4, 5))

    # physicians per shift, sized stream by stream at the base arrival rate; every
    # configuration gets the same total (the pooled configuration pools the two)
    phys, phys_low = [], []
    shift_rates = []
    for i, h0 in enumerate(shifts):
        h1 = shifts[i + 1] if i + 1 < len(shifts) else 24
        lam = base_rate / MIN_PER_DAY * prof[h0:h1].mean()      # patients per minute
        shift_rates.append((lam, h1 - h0))
        c_hi = max(1, int(math.ceil(lam * wl_hi["physician"] / rho_med)))
        c_lo = max(1, int(math.ceil(lam * wl_low["physician"] / rho_med)))
        phys.append(c_hi + c_lo)
        phys_low.append(c_lo)
    offered = sum(l * wl["physician"] * hrs for l, hrs in shift_rates)
    capacity = sum(c * hrs for c, (_, hrs) in zip(phys, shift_rates))
    rho0 = offered / capacity
    rate_medium = base_rate * rho_med / rho0

    # treatment spaces (stream by stream) and diagnostics, sized at the medium arrival rate
    lam_med = rate_medium / MIN_PER_DAY
    ss = v(cs["space_sizing_utilisation"])
    n_hi = max(1, int(math.ceil(lam_med * wl_hi["space"] / ss)))
    n_lo = max(1, int(math.ceil(lam_med * wl_low["space"] / ss)))
    n_space = n_hi + n_lo
    diag_cap = {s: max(1, int(math.ceil(lam_med * wl[s] / v(cs["diagnostic_target_utilisation"]))))
                for s in DIAG_STAGES}

    n_second, phys_second = 0, []
    if config != "POOLED":
        n_second = n_lo
        if config == "FASTTRACK":
            phys_second = list(phys_low)

    # actual load level: the arrival rate is scaled so that the busiest physician pool
    # (under the reference CTAS 4-5 split for FASTTRACK) has the target offered load
    target = v(P["load_levels"])[load] if isinstance(load, str) else float(load)
    if config == "FASTTRACK":
        cap_lo = sum(c * hrs for c, (_, hrs) in zip(phys_low, shift_rates))
        rho_hi = sum(l * wl_hi["physician"] * hrs for l, hrs in shift_rates) / (capacity - cap_lo)
        rho_lo = sum(l * wl_low["physician"] * hrs for l, hrs in shift_rates) / cap_lo
        rho_cfg = max(rho_hi, rho_lo)
    else:
        rho_cfg = rho0
    rate = base_rate * target / rho_cfg
    offered_load = target

    # triage nurses: peak hour at the high load level
    rate_high = rate_medium * v(P["load_levels"])["high"] / rho_med
    peak = rate_high / MIN_PER_DAY * prof.max()
    nurses = int(math.ceil(peak * P["durations"]["triage"]["mean"] / v(cs["triage_target_utilisation_peak"])))

    if seed_parts is None:
        f = design()["factors"]
        seed_parts = (f["acuity"].index(acuity), f["load"].index(load),
                      f["diagnostics"].index(diagnostics), f["boarding"].index(boarding))
    name = lambda x, fmt: x if isinstance(x, str) else fmt.format(x)
    return EDType(label or name(acuity if isinstance(acuity, str) else "custom", "{}"),
                  name(load, "{:.3f}"), config, name(boarding, "{:.0f}"), name(diagnostics, "{:+.2f}"),
                  visits_per_day=rate, load_multiplier=rate / base_rate,
                  physicians_by_shift=phys, second_physicians_by_shift=phys_second,
                  spaces_total=n_space, spaces_second=n_second, diag_capacity=diag_cap,
                  triage_nurses=nurses, offered_physician_load=offered_load,
                  mix=tuple(mix_value(acuity)), diag_shift=diag_shift_value(diagnostics),
                  board_mean=boarding_mean_value(boarding), perturb=dict(perturb or {}),
                  seed_parts=tuple(int(x) for x in seed_parts))


def core_ed_types() -> list:
    f = design()["factors"]
    return [build_ed_type(a, l, c) for a in f["acuity"] for l in f["load"] for c in f["config"]]
