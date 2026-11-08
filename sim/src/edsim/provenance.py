"""Provenance record written with every results file."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess

from . import __version__
from . import params as P


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=P.REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "sim"], cwd=P.REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def config_hash() -> str:
    h = hashlib.sha256()
    for name in ("parameters.json", "policies.json", "design.json"):
        h.update((P.CONFIG_DIR / name).read_bytes())
    return h.hexdigest()[:16]


def record(experiment: str, **extra) -> dict:
    rc = P.parameters()["run_control"]
    rec = {
        "experiment": experiment,
        "edsim_version": __version__,
        "git_commit": git_commit(),
        "config_sha256_16": config_hash(),
        "base_seed": P.v(rc["base_seed"]),
        "warmup_days": P.v(rc["warmup_days"]),
        "observation_days": P.v(rc["observation_days"]),
        "max_cooldown_days": P.v(rc["max_cooldown_days"]),
        "python": platform.python_version(),
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    rec.update(extra)
    return rec


def write(path, rec: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
