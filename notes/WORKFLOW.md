# Workflow and to-do list

> **Status 2026-11-08:** Phases 0-7 were run straight through to a first complete draft (see DECISIONS.md, 2026-11-08). Gates 1, 2, 4, 6 and 7 still need co-author review; GitHub repository and TEM submission (Phase 8) are pending.

Phases run in order. Each phase ends with a **gate**: a check that must pass before the next phase starts. Tick items as they are done, and log decisions in `DECISIONS.md`.

---

## Phase 0: Set-up (week 1)
- [x] Project folder, brief, decisions log, TEM template, shared bibliography
- [ ] Open a new Cowork project on this folder; start a new chat from `PROJECT_BRIEF.md`
- [ ] Read the TEM author guidelines: length, review type (single- or double-blind), data/code policy, scope fit
- [ ] Create a private GitHub repository (public at submission); add `.gitignore`, licence, README
- [ ] Python environment: `pyproject.toml`, pinned versions, `pytest`, `ruff`

**Gate 0:** the repository builds and an empty test suite runs.

## Phase 1: Literature and positioning (weeks 1–3)
- [ ] Search and screen: ED streaming / fast track / split flow; prioritization rules (FIFO, CTAS priority, APQ, shortest-service, prediction-based); ED simulation across ED types; policy-fit or contingency studies; ED KPI sets
- [ ] Reading notes in `lit/notes/` (one file per paper: setting, ED type, policies, KPIs, findings)
- [ ] Gap table: which ED types × policies × KPIs earlier studies covered
- [ ] Add every reference to `../ED Reference Library/ed_library.bib` (new key format)
- [ ] Draft research questions and contributions (one page)

**Gate 1:** the co-authors agree on the research questions, the contributions and the gap.

## Phase 2: Model specification on paper, before any code (weeks 3–5)
- [ ] **ED types:** factors and levels (acuity mix, load, configuration, boarding, diagnostic intensity), with a published source or "assumption" recorded for every parameter
- [ ] **Patient generator:** true attributes, CTAS-conditional pathway distribution (from the Value in Health table), activity durations, predicted attributes via a confusion matrix with error level α
- [ ] **Process model:** stages, resources and queues (simplified from the Markov chain); boarding; optionally patients who leave without being seen
- [ ] **Policies:** queue rules and bed-assignment rules, each defined once (name, parameters, tie-break)
- [ ] **KPIs:** one exact definition each (formula, time stamps, which patients are included); RATP and AWT fixed here
- [ ] **Fidelity propositions:** when the similarity ranking equals a classical rule
- [ ] Write all of this in `notes/MODEL_SPEC.md`; it becomes the paper's model section and the STRESS-DES entries

**Gate 2:** the specification is reviewed and frozen (version 1.0). Later changes go through `DECISIONS.md`.

## Phase 3: Build the simulation (weeks 5–9)
- [ ] Configuration schema (typed), one file per ED type
- [ ] Patient generator (seeded; writes the patient list to file)
- [ ] Discrete-event engine: arrivals, queues, resources, pathways, boarding
- [ ] Policy module (all rules behind one interface)
- [ ] Event log: one record per patient per stage
- [ ] KPI module: every KPI computed from the event log only
- [ ] Provenance written with every run: git commit, configuration, seeds, replications, warm-up, run length
- [ ] **Tests:**
  - unit tests per queue rule on small deterministic cases;
  - KPI functions against hand-computed examples;
  - patient conservation;
  - time stamps in order;
  - same policy + same seed gives identical results.

**Gate 3:** all tests pass in continuous integration (GitHub Actions).

## Phase 4: Verification and validation (weeks 9–11)
- [ ] Warm-up chosen by Welch's or MSER method; cool-down; run length
- [ ] Replications chosen by a CI half-width rule
- [ ] Validation against published aggregate ED statistics and the Value in Health pathway/Markov frequencies
- [ ] **Policy fidelity tests:**
  - decision-level: agreement rate and Kendall's τ between the similarity rule and each reference rule;
  - outcome-level: KPIs equivalent within CIs.
- [ ] Fill in the STRESS-DES checklist as each item is settled

**Gate 4:** validation and fidelity results are reviewed; the model is frozen (version 1.0, git tag).

## Phase 5: Experiments (weeks 11–14)
- [ ] Experimental design: factorial or Latin hypercube over ED types × policies × error level α
- [ ] One script per experiment; results stored as raw files with provenance
- [ ] Main runs: policy comparison per ED type
- [ ] Prediction-value runs: α sweep
- [ ] Sensitivity runs: KPI targets, policy weights, load
- [ ] Automated consistency checks on every results file (overall values within group ranges; same policy gives the same numbers across experiments)

**Gate 5:** all runs complete from the frozen model tag; checks pass.

## Phase 6: Analysis and outputs (weeks 14–16)
- [ ] One script, `make_outputs.py`, builds every table (`.tex`), figure and `numbers.tex` (macros) from raw results
- [ ] Tables: one per KPI, policies as columns, CIs and paired differences
- [ ] Figures:
  - heatmap of best policy by ED type;
  - regret of one-size-fits-all;
  - efficiency–fairness frontier;
  - tail risk;
  - value of prediction accuracy.
- [ ] Decision tree (metamodel) mapping ED characteristics to a recommended policy
- [ ] Interpretation notes: what holds across ED types, what changes, and why

**Gate 6:** a co-author re-runs `make_outputs.py` from a clean clone and gets identical files.

## Phase 7: Writing (weeks 16–20)
- [ ] Introduction and literature review (from Phase 1)
- [ ] Framework section (from `MODEL_SPEC.md`)
- [ ] Experimental design, results (text cites only `numbers.tex` macros), managerial implications, limitations, conclusion
- [ ] Abstract and highlights last
- [ ] Plain academic tone; claims match the results, including tail effects and limits

**Gate 7:** internal review by all co-authors; a number audit (`grep` for any digit typed in the results text that is not a macro).

## Phase 8: Submission (weeks 20–21)
- [ ] Final run from the tagged release; rebuild all outputs
- [ ] STRESS-DES checklist as supplementary material
- [ ] Repository: public release plus a Zenodo DOI (or an anonymized branch if TEM is double-blind; no names in CITATION/LICENSE)
- [ ] Cover letter, suggested reviewers, data and code availability statement
- [ ] Submit to IEEE TEM

---

### Standing rules (every phase)
- No hand-typed results; regenerate, don't edit.
- Every parameter has a source; every decision goes in `DECISIONS.md`.
- Commit often, with clear messages; tag the model versions used for results.
