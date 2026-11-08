"""Discrete-event engine for the ED model.

Patient journey: arrival -> triage (nurse) -> registration (delay) -> routing to an
area -> treatment space -> physician initial assessment (AS) -> diagnostic/consult
steps of the patient's pathway (LB, XR, CT, DR, each with its own queue) ->
physician reassessment (RA, only if there were steps) -> disposition decision ->
discharge process (DD) or boarding until an inpatient bed is available.

Areas hold their spaces either for the whole stay ('bed') or only during physician
encounters ('encounter', the vertical area of split flow). Every queue except
triage is ordered by the policy's queue rule; CTAS 1 is always served first.

The engine writes an event log: one record per patient and activity
(patient, stage, resource, t_request, t_start, t_end). All KPIs are computed from
this log by kpis.py.
"""
from __future__ import annotations

import heapq
import math
from itertools import count

from . import params as P
from .policies import FCFS

# event types
ARR, TRI_END, REG_END, PHYS_END, STEP_END, END_END, SHIFT = range(7)


class RuleQueue:
    """One heap per CTAS level; CTAS 1 first; heads compared with the rule's dynamic key."""
    __slots__ = ("rule", "heaps", "n", "name")

    def __init__(self, rule, name=""):
        self.rule = rule
        self.heaps = [None, [], [], [], [], []]
        self.n = 0
        self.name = name

    def push(self, pid, ctas, code, t_enter, seq, payload):
        sk = self.rule.static_key(code, ctas, t_enter, seq)
        heapq.heappush(self.heaps[ctas], (sk, pid, payload))
        self.n += 1

    def best_level(self, now):
        if self.heaps[1]:
            return 1
        best, best_c = None, 0
        dk = self.rule.dyn_key
        for c in (2, 3, 4, 5):
            h = self.heaps[c]
            if h:
                k = dk(h[0][0], c, now)
                if best is None or k < best:
                    best, best_c = k, c
        return best_c

    def pop(self, now):
        c = self.best_level(now)
        self.n -= 1
        sk, pid, payload = heapq.heappop(self.heaps[c])
        return pid, payload

    def entries(self):
        for c in (1, 2, 3, 4, 5):
            for sk, pid, payload in self.heaps[c]:
                yield c, sk, pid, payload


class Resource:
    __slots__ = ("name", "cap", "busy", "queue")

    def __init__(self, name, cap, queue):
        self.name = name
        self.cap = cap
        self.busy = 0
        self.queue = queue


class Engine:
    def __init__(self, ed, patients, queue_rule, routing_rule=None, alpha=1.0,
                 shadow=None, shadow_routing=None, tau_every=5):
        rc = P.parameters()["run_control"]
        self.warm = P.v(rc["warmup_days"]) * P.MIN_PER_DAY
        self.obs_end = self.warm + P.v(rc["observation_days"]) * P.MIN_PER_DAY
        self.ed = ed
        self.pts = patients
        self.alpha = alpha
        self.rule = queue_rule
        self.routing = routing_rule
        self.codes = [p.pred[alpha] for p in patients]
        self.now = 0.0
        self.events = []
        self.seq = count()
        self.log = []                       # (pid, stage, resource, t_req, t_start, t_end)
        self.n = len(patients)
        self.area = [None] * self.n         # current area of each patient
        self.routed_second = [False] * self.n
        self.space_start = [0.0] * self.n
        self.space_req = [0.0] * self.n
        self.req = [0.0] * self.n
        self.start = [0.0] * self.n
        self.step_i = [0] * self.n
        self.observed_left = sum(1 for p in patients if self.warm <= p.arrival < self.obs_end)
        self.depart = [math.nan] * self.n
        # fidelity logging
        self.shadow = shadow
        self.shadow_routing = shadow_routing
        self.tau_every = tau_every
        self.fid = {"n": 0, "agree": 0, "tau": []}
        self.fid_route = {"n": 0, "agree": 0}
        self._dispatch_count = 0

        cfg = P.design()["configurations"][ed.config]
        self.second = None
        self.second_holds = "bed"
        self.overflow = False
        self.dedicated = False
        if ed.config != "POOLED":
            self.second = cfg["areas"][1]
            self.second_holds = cfg["second_area_holds"]
            self.dedicated = cfg["dedicated_physicians"]
            self.overflow = bool(cfg.get("overflow_to_main", False))

        mk = lambda nm: RuleQueue(self.rule, nm)
        self.triage = Resource("TRIAGE", ed.triage_nurses, RuleQueue(FCFS("TRIAGE-FCFS"), "TRIAGE"))
        n_second = ed.spaces_second if self.second else 0
        self.spaces = {"MAIN": Resource("MAIN", ed.spaces_total - n_second, mk("SPACE-MAIN"))}
        if self.second:
            self.spaces[self.second] = Resource(self.second, n_second, mk("SPACE-" + self.second))
        self.phys = {"MAIN": Resource("PHYS-MAIN", 0, mk("PHYS-MAIN"))}
        if self.second and self.dedicated:
            self.phys[self.second] = Resource("PHYS-" + self.second, 0, mk("PHYS-" + self.second))
        self.diag = {s: Resource(s, ed.diag_capacity[s], mk(s)) for s in P.DIAG_STAGES}
        # physician shifts
        self.shift_hours = P.v(P.parameters()["capacity_sizing"]["shifts_start_hours"])
        self._set_shift(0)

    # ------------------------------------------------------------ helpers
    def _pool(self, pid):
        a = self.area[pid]
        return self.phys[a] if a in self.phys else self.phys["MAIN"]

    def _schedule(self, t, etype, pid, data=None):
        heapq.heappush(self.events, (t, next(self.seq), etype, pid, data))

    def _set_shift(self, idx):
        main = self.ed.physicians_by_shift[idx]
        if self.second and self.dedicated:
            sec = self.ed.second_physicians_by_shift[idx]
            self.phys[self.second].cap = sec
            self.phys["MAIN"].cap = main - sec
        else:
            self.phys["MAIN"].cap = main

    def _shift_events(self, horizon):
        day = 0
        while day * P.MIN_PER_DAY < horizon:
            for i, h in enumerate(self.shift_hours):
                t = day * P.MIN_PER_DAY + h * 60
                if t > 0:
                    self._schedule(t, SHIFT, -1, i)
            day += 1

    def _enqueue(self, res, pid, payload):
        p = self.pts[pid]
        self.req[pid] = self.now
        res.queue.push(pid, p.ctas, self.codes[pid], self.now, pid, payload)
        self._dispatch(res)

    def _pop(self, res):
        q = res.queue
        if self.shadow is not None and q.n >= 2 and q.name != "TRIAGE":
            self._fidelity(q)
        return q.pop(self.now)

    def _fidelity(self, q):
        """Compare the choice of the running rule with the shadow rule on this queue."""
        now = self.now
        ents = list(q.entries())
        def keys(rule, use_existing):
            out = []
            for c, sk, pid, _ in ents:
                if use_existing:
                    k = rule.dyn_key(sk, c, now)
                else:
                    p = self.pts[pid]
                    sk2 = rule.static_key(self.codes[pid], c, sk[-2], sk[-1])
                    k = rule.dyn_key(sk2, c, now)
                out.append((0 if c == 1 else 1,) + k)
            return out
        ka = keys(self.rule, True)
        kb = keys(self.shadow, False)
        ia = min(range(len(ents)), key=ka.__getitem__)
        ib = min(range(len(ents)), key=kb.__getitem__)
        self.fid["n"] += 1
        self.fid["agree"] += int(ia == ib)
        self._dispatch_count += 1
        if len(ents) >= 3 and self._dispatch_count % self.tau_every == 0:
            ra = _ranks(ka)
            rb = _ranks(kb)
            self.fid["tau"].append(_kendall_tau(ra, rb))

    def _dispatch(self, res):
        while res.busy < res.cap and res.queue.n:
            pid, payload = self._pop(res)
            self._start(res, pid, payload)
        # overflow: an idle Main space may take a waiting fast-track patient
        if (self.overflow and res.name == "MAIN" and res.busy < res.cap
                and res.queue.n == 0 and self.spaces[self.second].queue.n):
            sec = self.spaces[self.second]
            while res.busy < res.cap and sec.queue.n:
                pid, payload = self._pop(sec)
                self.area[pid] = "MAIN"
                self._start(res, pid, payload)
        elif self.overflow and res.name == self.second and res.queue.n:
            main = self.spaces["MAIN"]
            if main.busy < main.cap and main.queue.n == 0:
                self._dispatch(main)

    # ------------------------------------------------------------ starting activities
    def _start(self, res, pid, payload):
        res.busy += 1
        now = self.now
        p = self.pts[pid]
        name = res.name
        if name == "TRIAGE":
            self.start[pid] = now
            self._schedule(now + p.d_triage, TRI_END, pid)
        elif name in self.spaces:
            self.space_start[pid] = now
            self.space_req[pid] = self.req[pid]
            if payload == "BOARD":
                self.start[pid] = now
                self._schedule(now + p.d_end, END_END, pid, "BOARD")
            else:
                self._enqueue(self._pool(pid), pid, payload)
        elif name.startswith("PHYS"):
            self.start[pid] = now
            d = p.d_as if payload == "AS" else p.d_ra
            self._schedule(now + d, PHYS_END, pid, (payload, name))
        else:  # diagnostic / consult
            self.start[pid] = now
            self._schedule(now + p.d_steps[payload], STEP_END, pid, payload)

    def _release_space(self, pid):
        a = self.area[pid]
        res = self.spaces[a]
        res.busy -= 1
        self.log.append((pid, "SPACE", a, self.space_req[pid], self.space_start[pid], self.now))
        self._dispatch(res)

    def _holds_through_stay(self, pid):
        return self.area[pid] == "MAIN" or self.second_holds == "bed"

    # ------------------------------------------------------------ run
    def run(self):
        horizon = self.pts[-1].arrival if self.pts else 0.0
        for p in self.pts:
            self._schedule(p.arrival, ARR, p.pid)
        self._shift_events(horizon + 30 * P.MIN_PER_DAY)
        ev = self.events
        while ev and self.observed_left > 0:
            t, _, etype, pid, data = heapq.heappop(ev)
            self.now = t
            if etype == ARR:
                self._enqueue(self.triage, pid, None)
            elif etype == TRI_END:
                self.triage.busy -= 1
                self.log.append((pid, "TRIAGE", "TRIAGE", self.req[pid], self.start[pid], t))
                self._dispatch(self.triage)
                self._schedule(t + self.pts[pid].d_reg, REG_END, pid)
            elif etype == REG_END:
                self.log.append((pid, "REG", "", t - self.pts[pid].d_reg, t - self.pts[pid].d_reg, t))
                self._route(pid)
            elif etype == PHYS_END:
                task, rname = data
                res = self.phys[rname[5:]] if rname[5:] in self.phys else self.phys["MAIN"]
                res.busy -= 1
                self.log.append((pid, task, rname, self.req[pid], self.start[pid], t))
                self._dispatch(res)
                if not self._holds_through_stay(pid):
                    self._release_space(pid)
                if task == "AS" and self.pts[pid].steps:
                    self.step_i[pid] = 0
                    self._start_step(pid)
                else:
                    self._decide(pid)
            elif etype == STEP_END:
                i = data
                p = self.pts[pid]
                res = self.diag[p.steps[i]]
                res.busy -= 1
                self.log.append((pid, p.steps[i], p.steps[i], self.req[pid], self.start[pid], t))
                self._dispatch(res)
                if i + 1 < len(p.steps):
                    self.step_i[pid] = i + 1
                    self._start_step(pid)
                elif self._holds_through_stay(pid):
                    self._enqueue(self._pool(pid), pid, "RA")
                else:
                    self._enqueue(self.spaces[self.area[pid]], pid, "RA")
            elif etype == END_END:
                stage = data
                self.log.append((pid, stage, "", self.start[pid], self.start[pid], t))
                if stage == "BOARD" or self._holds_through_stay(pid):
                    self._release_space(pid)
                self._depart(pid)
            elif etype == SHIFT:
                self._set_shift(data)
                for r in self.phys.values():
                    self._dispatch(r)
        return self

    def _route(self, pid):
        p = self.pts[pid]
        code = self.codes[pid]
        second = False
        if self.second is not None and self.routing is not None:
            second = self.routing.to_second(p.ctas, code)
            if self.shadow_routing is not None:
                self.fid_route["n"] += 1
                self.fid_route["agree"] += int(second == self.shadow_routing.to_second(p.ctas, code))
        a = self.second if second else "MAIN"
        self.area[pid] = a
        self.routed_second[pid] = second
        self._enqueue(self.spaces[a], pid, "AS")

    def _start_step(self, pid):
        i = self.step_i[pid]
        self._enqueue(self.diag[self.pts[pid].steps[i]], pid, i)

    def _decide(self, pid):
        p = self.pts[pid]
        self.log.append((pid, "DECISION", "", self.now, self.now, self.now))
        if p.admit and not self._holds_through_stay(pid):
            # vertical patient: boards in a Main bed
            self.area[pid] = "MAIN"
            self._enqueue(self.spaces["MAIN"], pid, "BOARD")
        else:
            self.start[pid] = self.now
            self._schedule(self.now + p.d_end, END_END, pid, "BOARD" if p.admit else "DD")

    def _depart(self, pid):
        self.depart[pid] = self.now
        a = self.pts[pid].arrival
        if self.warm <= a < self.obs_end:
            self.observed_left -= 1

    # ------------------------------------------------------------ capacity record for utilisation
    def capacity_minutes(self, t0, t1) -> dict:
        """Capacity-minutes of each resource in [t0, t1)."""
        out = {}
        sp = self.spaces
        for a, r in sp.items():
            out[a] = r.cap * (t1 - t0)
        for s, r in self.diag.items():
            out[s] = r.cap * (t1 - t0)
        # physicians follow the shift schedule
        hrs = self.shift_hours + [24]
        phys = {k: 0.0 for k in self.phys}
        day = int(t0 // P.MIN_PER_DAY)
        while day * P.MIN_PER_DAY < t1:
            for i in range(len(self.shift_hours)):
                a0 = max(t0, day * P.MIN_PER_DAY + hrs[i] * 60)
                a1 = min(t1, day * P.MIN_PER_DAY + hrs[i + 1] * 60)
                if a1 > a0:
                    tot = self.ed.physicians_by_shift[i]
                    if self.second and self.dedicated:
                        sec = self.ed.second_physicians_by_shift[i]
                        phys[self.second] += sec * (a1 - a0)
                        phys["MAIN"] += (tot - sec) * (a1 - a0)
                    else:
                        phys["MAIN"] += tot * (a1 - a0)
            day += 1
        for k, val in phys.items():
            out["PHYS-" + k] = val
        return out


def _ranks(keys):
    order = sorted(range(len(keys)), key=keys.__getitem__)
    r = [0] * len(keys)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def _kendall_tau(ra, rb):
    """Kendall's tau-a between two strict rankings (no ties: keys end in unique ids)."""
    n = len(ra)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (ra[i] - ra[j]) * (rb[i] - rb[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = n * (n - 1) / 2
    return (conc - disc) / tot if tot else 1.0
