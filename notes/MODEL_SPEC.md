# Model specification (version 1.0, 2026-11-08)

This is the frozen specification of the simulation used in the paper. Every number
mentioned here lives in `sim/configs/*.json` with its source; this file explains the
logic. Changes after version 1.0 go through `DECISIONS.md`.

---

## 1. Purpose and outputs

Compare patient-streaming policies (queue rule + routing rule) across ED types; measure
the loss from using one policy everywhere (regret); measure how good predictions must be;
check whether similarity-based rules apply the policies they are meant to represent.

## 2. ED types (designed experiment)

| Factor | Levels | Notes |
|---|---|---|
| Acuity mix | tertiary, community, urgent-care-like | CTAS shares inside the between-site ranges of Holroyd et al. (2016); community = their pooled distribution |
| Load | low, medium, high | offered load 0.70 / 0.80 / 0.90 of the busiest physician pool, reached by scaling the arrival rate |
| Configuration | POOLED, FASTTRACK, SPLITFLOW | see Section 4 |
| Boarding | low (core), high | mean decision-to-departure 180 / 480 min, lognormal, CV 1 |
| Diagnostic intensity | light, typical (core), heavy | shift of the pathway tilt (Section 3) |

Core design: full factorial acuity × load × configuration (27 ED types) with low boarding
and typical diagnostics. Sensitivity: boarding high, diagnostics light, diagnostics heavy,
one at a time, on the nine medium-load types.

**Capacity sizing** (code: `params.build_ed_type`). The base volume is 220 visits/day with
a fixed hourly arrival profile. Two patient streams are sized separately: CTAS 1–3 and
CTAS 4–5.
- Physicians: three 8-hour shifts; each stream gets ceil(rate × expected physician work / 0.80) physicians per shift.
- Treatment spaces: ceil(rate × expected service-only holding time / 0.60) per stream, at the medium-load rate.
- Diagnostics: offered load 0.60 at the medium-load rate.
- Triage nurses: offered load 0.70 at the peak hour of the high-load rate.

Every configuration of an acuity mix has the same totals: POOLED pools the two streams;
FASTTRACK and SPLITFLOW give the CTAS 4–5 stream's spaces to the second area, and
FASTTRACK also gives it that stream's physicians. The arrival rate is then scaled so that
the busiest physician pool has the target offered load (under CTAS 4–5 streaming for
FASTTRACK). An ED with a dedicated fast track therefore handles fewer visits at the same
load label; this is reported.

## 3. Patient generator (`generator.py`)

For each ED type and replication, one patient list:
1. Arrivals: non-homogeneous Poisson (thinning) with the hourly profile; horizon = warm-up + observation + 10 days of cool-down.
2. CTAS from the acuity mix; admission with the CTAS-specific probability (midpoints of the Holroyd et al. ranges).
3. A whole pathway drawn from the 11 published pathways (Rismanchian et al., 2023) that end in the patient's disposition, with weight
   `freq × exp((tilt_CTAS + shift_diag) × number of diagnostic/consult steps)`. Tilt is 0 for CTAS 3 with typical diagnostics, which reproduces the published within-disposition frequencies. The "other" 21.4% of visits is represented by renormalising over the 11 pathways.
4. Durations (lognormal, mean and CV in `parameters.json`): triage, registration, physician initial assessment (by CTAS), reassessment, LB, XR, CT, DR, discharge process, boarding.
5. True code (six attributes): A CTAS; B service class (five equal-frequency classes of total treatment time, cut points from the community reference sample; 1 = very long … 5 = very short); C admission (1 admit, 2 discharge); D diagnostics (1 any of LB/XR/CT, 2 none); E resource use (1 = 3+ steps, 2 = 2, 3 = 1, 4 = none); F consult (1 DR, 2 none).
6. Predicted code: A is observed. For B–F, M(α) = I + α(M_ref − I). M_ref has the reference accuracy on the diagonal; errors are spread over other classes with weight 0.35^(distance−1). One uniform draw per patient and attribute is shared by all α (common random numbers across α). Attributes are predicted independently.

## 4. Process model (`engine.py`)

Arrival → triage (nurse; FCFS, CTAS 1 first) → registration (delay) → routing → treatment
space → physician initial assessment (AS) → the pathway's steps in order (LB, XR, CT, DR;
each a multi-server resource with its own queue) → physician reassessment (RA; only if
there were steps) → decision → discharge process (DD) or boarding (AD) → departure.

- **POOLED**: one area of beds; one physician pool.
- **FASTTRACK**: Main and FT areas with their own beds and physicians. A patient waiting for an FT space may take an idle Main space when no Main patient is waiting (then served by Main physicians).
- **SPLITFLOW**: Main beds plus a vertical area of chairs held only during physician encounters (AS and RA); diagnostics are waited for without a chair; one shared physician pool; an admitted vertical patient boards in a Main bed (joins the Main bed queue).
- Beds in Main/FT are held from arrival in the space until departure (including boarding).
- Physicians: shift capacity changes are non-preemptive. A patient may see any physician of the area's pool for reassessment.
- Not modelled: patients who leave without being seen, physician hand-overs, nurses other than triage, time-varying staffing within a shift.

## 5. Policies (`policies.py`, `configs/policies.json`)

Queue rules (applied at every space, physician and diagnostic queue; CTAS 1 always first;
ties first-come-first-served, then patient id):
FCFS; CTAS; APQ (priority = wait / CTAS target); CTAS-SPT (CTAS, then shortest predicted
service); CTAS-PD (CTAS, then predicted discharge first); SIM (Offodile similarity to
target 1-5-2-2-4-2 with weights 0.40, 0.30, 0.10, 0.10, 0.05, 0.05); SIM-AGE (SIM + 0.10 per
elapsed CTAS target time).

Routing rules (two-area configurations): STREAM-CTAS (CTAS 4–5 to the second area);
STREAM-PRED (CTAS 3–5 with predicted discharge and predicted service class 4–5);
STREAM-SIM (CTAS 3–5 to the area with the more similar target: Main 1-1-1-1-1-1, second
5-5-2-2-4-2; ties to Main).

A policy is a (queue rule, routing rule) pair: 7 for POOLED, 7 × 3 = 21 otherwise.
Default ("common practice"): CTAS with STREAM-CTAS. Similarity scores are rounded to 1e-9
before comparison, so equal scores tie exactly and fall to FCFS.

## 6. Policy fidelity of the similarity representation

Notation: code p, target q, weights w ≥ 0, range R_k = n_k − 1,
S(p) = Σ_k w_k (1 − |p_k − q_k| / R_k); the queue serves the largest S.

**Proposition 1 (linear index).** If every weighted attribute has its target at an end of
its range, S is affine in p, so the rule is a static linear priority index that is
monotone in every attribute. If a weighted target is interior (e.g. B = 3 in the IISE
main-pod code), S is not monotone in that attribute, and the rule cannot equal any rule
that is monotone in it.

**Proposition 2 (lexicographic dominance).** Targets at ends of their ranges; attributes
ordered k = 1, …, K. If w_k / R_k > Σ_{j>k} w_j for every k with w_k > 0, the
similarity order equals the lexicographic order of the distances |p_k − q_k|, with FCFS
ties. Hence weights (1, 0, …) give strict CTAS priority; w_A/4 > w_B with q_B = 5 gives
CTAS-SPT; w_A/4 > w_C with q_C = 2 gives CTAS-PD. If instead
w_k / R_k ≤ Σ_{j>k} w_j for some k and all code combinations can occur, there are two
patients whom S orders differently (or ties, so FCFS decides) from the lexicographic rule.
Proof sketch: for codes first differing at k by one step, the change in term k is at
least w_k/R_k and later terms change by at most Σ_{j>k} w_j; for the converse take the
patient closer at k by one step but farthest from the target on every later attribute.

**Proposition 3 (two-target routing).** With targets at opposite ends on every weighted
attribute, S_second − S_main = Σ_k σ_k (w_k/R_k)(2p_k − n_k − 1), σ_k = ±1: routing is a
linear threshold rule whose threshold passes through the middle of the code ranges. Weights
on CTAS only give CTAS streaming exactly (tie at CTAS 3 goes to Main). A rule whose
boundary is off-centre, such as STREAM-PRED (to the second area only if predicted
discharge and service class ≥ 4), cannot be represented: with C = 2, B = 3 the C-term is
positive and the B-term zero.

Remarks: with all weights zero, SIM is FCFS (all ties); SIM-AGE with zero weights is APQ.

**Tests.** Propositions are checked by exhaustive enumeration in `tests/test_edsim.py`;
decision-level and outcome-level fidelity are measured in E3 (agreement rate of first
choices, Kendall's τ between queue orders at every fifth dispatch with ≥ 3 waiting,
identical event logs, paired KPI differences).

## 7. KPIs (`kpis.py`), from the event log only

Observed patients: arrival in the observation window; every observed patient is followed
to departure (cool-down), so nothing is truncated.
- Door-to-triage; time to treatment space (TTB: door to first space); door-to-physician
  (PIA: door to start of AS); LOS (door to departure), separately for discharged and admitted;
  boarding (decision to departure). Mean, median, 90th and 95th percentiles, maximum;
  overall and by CTAS.
- PTP_l: share of CTAS-l patients whose door-to-physician time exceeds the CTAS target
  (15, 15, 30, 60, 120 min; CTAS 1 "immediate" adapted to 15 min).
- RATP = Σ_l (1/l) PTP_l. AWT = Σ_l (1/l) mean_l max(0, door-to-physician − target_l) (minutes).
  Note: the IISE paper measured PTP on time-to-bed; here it is measured on time to physician,
  which is what the CTAS objectives define.
- Throughput per day; utilisation of every resource in the observation window; mean wait at each stage.

## 8. Run control

Warm-up 2 days (MSER-5: ≤ 15 h at high load), observation 28 days, cool-down until every
observed patient has left (≤ 10 days of further arrivals). Replications: 25 (main), 15
(fidelity, α sweep, sensitivity). Common random numbers: one patient list per
(ED type, replication) shared by all policies; seeds from a SeedSequence of
(base seed, factor levels, replication).

## 9. Assumptions without a published source

Base volume and hourly profile; all activity durations; diagnostic tilt and shift;
prediction reference accuracies and error decay; boarding levels (informed by provincial
medians); capacity-sizing targets (space sizing 0.60 calibrated); three equal shifts;
registration as a delay; independence of prediction errors across attributes; pathway
frequencies from one Korean academic ED (Rismanchian et al., 2023) combined with Canadian
case-mix figures.
