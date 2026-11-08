"""Run one ED type under one policy for one replication and return its KPIs."""
from __future__ import annotations

from .engine import Engine
from .generator import generate
from .kpis import compute_kpis
from .policies import make_queue_rule, make_routing_rule, policy_name


def run_policy(ed, patients, queue: str, routing: str | None, alpha: float = 1.0, **engine_kw):
    qr = make_queue_rule(queue)
    rr = make_routing_rule(routing) if (routing and ed.config != "POOLED") else None
    eng = Engine(ed, patients, qr, rr, alpha=alpha, **engine_kw).run()
    return eng


def kpi_row(ed, rep, queue, routing, alpha, eng) -> dict:
    row = {"acuity": ed.acuity, "load": ed.load, "config": ed.config, "boarding": ed.boarding,
           "diagnostics": ed.diagnostics, "rep": rep, "queue": queue, "routing": routing or "",
           "policy": policy_name(queue, routing if ed.config != "POOLED" else None), "alpha": alpha}
    row.update(compute_kpis(eng))
    return row


def run_ed_rep(ed, rep, policy_list, alpha=1.0, alphas=None):
    pts = generate(ed, rep, alphas=alphas)
    rows = []
    for q, r in policy_list:
        eng = run_policy(ed, pts, q, r, alpha)
        rows.append(kpi_row(ed, rep, q, r, alpha, eng))
    return rows
