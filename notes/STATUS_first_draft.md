# Status: first complete draft (2026-11-08)

All workflow phases were run straight through to a first complete draft (agreed with F.R.). Co-author gates (1, 2, 4, 6, 7) are still open. Every choice is logged in `notes/DECISIONS.md`; the model is specified in `notes/MODEL_SPEC.md`.

## What exists
- `sim/`: package `edsim` (generator, engine, policies, KPIs, provenance), 24 tests, configs with a source for every input.
- `experiments/`: E0 pilot, E1 validation, E2 main (27 ED types × all policies × 25 reps), E3 fidelity, E4 α sweep, E5 sensitivity; `check_results.py`, `make_outputs.py`, `audit_numbers.py`, `run_all.sh`. Final results regenerated from tag `v1.0.0`.
- `paper/`: `main_review.tex` (anonymised, double-spaced, for TEM), `main.tex` (two-column with authors), generated `numbers.tex`, tables and figures, STRESS-DES supplement, `make_package.sh` for the LaTeX zip.

## Main findings (values come from `paper/numbers.tex`)
1. Best policy differs by ED type, mainly with acuity mix and configuration (tertiary: SIM; urgent-care-like: CTAS-ordered rules).
2. The common default (CTAS order + CTAS fast track) has large regret (mean about 16% on RATP; above 10% in most ED types).
3. For any single measure, one well-chosen policy is near-best everywhere (about 3% RATP, under 1% LOS), but the policy changes with the measure; FCFS is best for CTAS 5 tail waits.
4. Routing matters at least as much as queue order; similarity routing wins by sending more patients to the second area. Prediction streaming overloads the Main area when capacity is split by acuity.
5. Gains of prediction-based policies hardly depend on prediction accuracy (rule structure, not information).
6. Fidelity: lexicographic weight conditions give exact equivalence (identical event logs); the SIM weights agree with CTAS-SPT on only about half of decisions despite nearly equal average outcomes.

## Open items for co-authors
- Title: findings support "one policy does not fit all" for the default policy and across objectives, less so for one objective. Keep, or retitle?
- Assumptions to review: activity durations, prediction accuracies, boarding levels, 220 visits/day, space sizing 0.60 (DECISIONS.md).
- Medium load is less congested at the median than Canadian averages (face validity section).
- Not modelled: LWBS, nurses beyond triage, hand-overs. Proposed extras in `claude/notes/sim_open_source_scan.md` (Ciw verification, LWBS) not adopted yet.
- Bibliography style: local builds use unsrtnat because IEEEtranN.bst could not be downloaded here; compile on Overleaf for IEEE style.
- GitHub: local repository with full history is in the project folder; push after the GitHub sign-in (private now; anonymised review branch needed for double-blind review).
