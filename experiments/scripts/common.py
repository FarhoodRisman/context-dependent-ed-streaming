"""Shared helpers for the experiment scripts: paths, parallel execution, output with provenance."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim" / "src"))

import pandas as pd  # noqa: E402

from edsim import params as P  # noqa: E402
from edsim import provenance  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
WORKERS = int(os.environ.get("EDSIM_WORKERS", os.cpu_count() or 1))


def outdir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def pmap(fn, tasks, workers=WORKERS):
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def save(df: pd.DataFrame, name: str, experiment: str, **extra):
    d = outdir(experiment)
    df.to_csv(d / f"{name}.csv.gz", index=False, compression="gzip")
    provenance.write(d / f"{name}.provenance.json",
                     provenance.record(experiment, file=f"{name}.csv.gz", rows=len(df), **extra))
    return d / f"{name}.csv.gz"
