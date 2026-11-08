# Project brief: One Policy Does Not Fit All

**Working title:** *One Policy Does Not Fit All: Matching Patient Streaming Policies to Emergency Department Types*
**Target journal:** IEEE Transactions on Engineering Management (TEM)
**Authors (tentative):** F. Rismanchian, F. Zaerpour, P. Hosseinzadeh Kassani
**Started:** 2026-11-07, while the IISE paper (UHSE-2026-0025) is under review.

Read this first in any new chat about this project. Decisions taken later go in `DECISIONS.md`.

---

## 1. Research question and framing

- **Main question:** which patient-streaming policy (prioritization rule plus bed assignment) works best for which type of ED, and how much does an ED lose by adopting a "one-size-fits-all" policy?
- **Assumption to test:** each ED needs a policy suited to its patient mix and configuration.
- **TEM angle (management):** a decision guide that tells ED managers which policy fits their ED. Possible framings are contingency/fit and the cost of a mismatch ("regret").
- **Gap:** the ED streaming and fast-track simulation literature consists mostly of single-site studies, and their conclusions on case mix disagree. A quick search (Nov 2026) found no head-to-head policy comparison across ED types. A proper literature review is still to do.
- **What is new compared with the IISE paper:**
  - many ED types and configurations instead of one ED;
  - bed assignment treated as a policy lever, not only queue order;
  - no machine learning;
  - a formal check that the similarity heuristic really applies the policy it is meant to represent.

## 2. Scope decisions so far

- **No ML.** Predictions are simulated: each patient has true attributes and predicted attributes, and the predicted ones come from the true ones through a confusion matrix whose error is scaled by α (as in the IISE robustness experiment). α = 0 gives the upper bound, "perfect information". This also answers how good predictions must be before prediction-based dispatching pays off.
- **Synthetic patients.** A generator creates the patient stream from an ED-type parameter file. Each replication generates one list, and every policy runs on the same list (common random numbers by construction).
- **Parameters from published aggregates only.** Use the literature, national ED reporting statistics and our own published aggregates, not WRHA patient-level data. Reusing WRHA-derived numbers beyond what is already published may need an ethics check under protocol H2022:128.
- **Open source.** A public GitHub repository, released at submission, with a separate anonymized review branch if TEM uses double-blind review (check TEM policy).

## 3. Process model (patient journey)

- **Base:** the Markov chain and pathway table from **Rismanchian et al. (2023), *Value in Health* 26(1):18–27, doi:10.1016/j.jval.2022.04.002** (bib key `Rismanchian2023_PatientJourney`).
  - Stages: triage (TR) → registration (RG) → physician assessment (AS) → diagnostic/consult loop {lab LB, X-ray XR, CT, consult DR} → discharge (DD) or admission (AD).
- **Pathways:** `lit/pathways_Rismanchian2023_VIH.csv` lists the 11 most frequent pathways, which cover 78.6% of visits. The single most common, TR→RG→AS→DD, is 24.3%.
- **Recommended use: sample a whole pathway per patient** from a distribution that depends on CTAS (with an "other" bucket), rather than step-by-step Markov transitions.
  - This keeps realistic sequences and avoids loops that never happen in practice.
  - Changing the pathway mix is how ED types differ in diagnostic intensity (e.g. an urgent-care-like ED has mostly AS→DD).
  - The Markov matrix stays as a validation check: the simulated transition frequencies should reproduce it.
- **Queues:** every stage that uses a resource (bed or treatment space, physician, lab, X-ray, CT, consultant) has its own queue, and the policy applies at each one.
- **Add:** boarding for admitted patients (decision → departure), which was missing in the IISE model. Optionally, patients who leave without being seen (a patience threshold), which was a limitation of the IISE paper.

## 4. ED types: a designed experiment

Factors (levels to be fixed):

| Factor | Example levels |
|---|---|
| Acuity mix | tertiary/high-acuity; mixed community; urgent-care-like (mostly CTAS 4–5) |
| Load | utilization low / medium / high (arrival-rate multiplier) |
| Configuration | single pooled area; two pods (Main + Urgent, as in IISE); fast track + main; split flow / vertical |
| Boarding | admission rate × boarding time (low/high) |
| Diagnostic intensity | pathway mix (light / typical / heavy) |

Run a full factorial on a few key factors, or a Latin hypercube design plus a metamodel.

## 5. Policies (two layers)

**Queue rules** (applied at each queue):
- first-come-first-served;
- strict CTAS priority (first-come-first-served within each level);
- APQ, the accumulating priority queue (Stanford, Taylor & Ziedins 2014);
- shortest predicted service first;
- predicted discharge first / predicted admission first;
- the similarity rule (Offodile code + target code + weights);
- each optionally with a waiting-time ageing term (the safeguard proposed in the IISE paper).

**Bed assignment / streaming:**
- pooled beds;
- CTAS-based streaming;
- prediction-based fast track (predicted discharge with short service);
- similarity-based routing to pods.

**Safety rule for all policies:** CTAS 1 is always served first.

## 6. Does the similarity heuristic really apply the policy? (planned contribution)

1. **Analytical:** state conditions on the weights and target code under which the similarity ranking is *exactly* a classical rule.
   - If w_CTAS is larger than the sum of all other weights, the ranking is lexicographic in CTAS, so it equals strict CTAS priority.
   - A weight/target choice that makes the score monotone in predicted service time gives shortest-predicted-service-first.
   - Write these up as propositions with proofs.
2. **Decision-level fidelity:** log every dispatch and compare it with what the reference rule would have chosen. Report the agreement rate and Kendall's τ between the two queue orderings.
3. **Outcome-level equivalence:** the KPIs under the heuristic and under the coded rule should fall within each other's confidence intervals (paired, using common random numbers).
- **Claim:** the similarity framework is one representation that reproduces classical rules exactly when the conditions hold, and approximates them with *measured* fidelity otherwise.
- **Tie-breaking must be explicit** (first-come-first-served). In the IISE paper, rounding differences of about 1e-17 in the weights changed tie-breaks, which caused the Table 18/19 mismatch.

## 7. KPIs (literature-based set; to refine)

- **Timing:**
  - door-to-triage;
  - time to physician initial assessment (PIA; benchmarked by CAEP, the Canadian emergency physicians' association);
  - time-to-bed (TTB);
  - decision-to-departure (boarding);
  - LOS for discharged and for admitted patients.
  - Report mean, median and 90th percentile (CAEP uses the 90th). Use the time-stamp definitions of the US ED Benchmarking Alliance (EDBA) consensus.
- **Target compliance:**
  - breach rate by CTAS level (PTP), measured against the CTAS response-time objectives (Bullard et al. 2017, key `Bullard2017_CTAS`; CTAS 1 at 15 minutes is a study-specific adaptation);
  - **RATP** = Σ_l (1/l)·PTP_l (acuity-weighted breach rate);
  - **AWT** = acuity-weighted lateness magnitude.
  - Report both, because they can disagree (an IISE finding). Fix the definitions in code and paper on day 1.
- **Flow and capacity:** throughput (patients/hour/day); utilization of beds, physicians, lab, X-ray and CT; queue lengths; waiting time at each stage; occupancy.
- **Equity and risk:** 95th-percentile and maximum waits by CTAS level; gaps between levels; left-without-being-seen (if modelled).
- **Robustness:** KPI change as prediction error α grows.
- Starting references: Sørup et al. 2013 (systematic review of ED performance measures); the CAEP position statement on overcrowding; EDBA definitions.

## 8. Simulation method: build in the IISE lessons from day 1

- Fill in the **STRESS-DES** checklist while building (template: `from_IISE/manuscript/STRESS_DES_checklist.tex`).
- **Warm-up:** chosen formally (Welch's or MSER), not a fixed one week.
- **Cool-down:** add one, so patients still in the ED at the end of a run aren't cut off (an IISE finding).
- **Replications:** set by a CI half-width rule, and stated in every table and figure caption.
- **Seeds and common random numbers:** fixed seeds, the same patient list across policies, and paired differences with confidence intervals.
- **KPIs from the event log:** compute every KPI, including any "Overall" row, by code from the patient-level event log. Never type numbers into tables by hand. (IISE Table 12's Overall TTB fell outside its group means.)
- **Verification:** unit tests for each queue rule on small deterministic scenarios; checks that patients are conserved and every time stamp falls in order.
- **Validation:** against published aggregate ED statistics, and against the Value in Health pathway and Markov frequencies.
- **One simulation engine.** IISE had Arena plus Python, which caused confusion about which model produced which number.

## 9. Results and analysis (improvements over IISE)

- One table per KPI with policies as columns, CIs and paired differences against the baseline (instead of one table per policy).
- A heatmap of the best policy by ED type.
- **Regret:** the KPI loss from using one policy everywhere versus the best policy per ED type. This is the headline managerial result.
- Efficiency-versus-fairness plots (Pareto) and tail-risk plots.
- A decision tree (metamodel) mapping ED characteristics to a recommended policy, as a decision aid for managers.

## 10. Code base: start clean (decided 2026-11-07)

- **Build a new package from scratch in `sim/`.**
  - Do not copy or fork code from the IISE model (`from_IISE/code/`) or from `../ed-digital-twin/`.
  - Do not reuse IISE results, parameters or numbers.
- **What may be used from IISE:** ideas and lessons only (this brief, the STRESS-DES checklist structure). `from_IISE/` is kept as read-only background and is never imported or copied from.
- **Parameters:** every input number is entered once, in a configuration file, with its source (citation or "assumption") recorded next to it.

## 10a. Number integrity: no hand-typed results

The IISE paper had several inconsistent numbers:
- the RATP values were mislabelled;
- the Overall TTB fell outside the range of its group means;
- Tables 18 and 19 disagreed for the same policy;
- a separate Arena model and Python model each produced numbers.

These rules prevent a repeat:

1. **One engine, one pipeline.** All results come from the single Python package. There is no second model.
2. **Every number in the paper is generated by code.**
   - Experiments write raw outputs: patient-level event logs and per-replication KPIs.
   - One script builds all tables (as `.tex`) and figures from those outputs, plus a `numbers.tex` file of LaTeX macros (e.g. `\newcommand{\RATPbaseCommunity}{1.23}`) that the text uses.
   - Nobody types a result into the manuscript by hand.
3. **One definition per KPI**, in one function, unit-tested on a hand-computed example. Aggregate rows such as "Overall" are computed from patient records, never from other table cells.
4. **Every results file records its provenance:** the git commit, the configuration file, seeds, number of replications, warm-up, run length and date. Tables and figures show which run produced them.
5. **Same policy means same object.** A policy is defined once (one name, one parameter set) and referenced everywhere, never rebuilt with re-computed weights. Ties are broken explicitly.
6. **Automated checks run before any result is used:**
   - unit tests and patient-conservation checks;
   - time stamps in order;
   - consistency checks (e.g. an overall mean must lie between its group means; the same policy and seed must give identical results across experiments).
7. **Regenerate rather than edit.** When something changes, re-run the pipeline and rebuild every table and figure, so text, tables and figures always come from the same run.

## 11. Folder map

```
One Policy Does Not Fit All/
  paper/        IEEE TEM template (template/), IEEEtran.cls, main.tex, ed_library.bib, figures/
  sim/          src/, tests/, configs/ed_types/
  experiments/  run scripts, results/
  lit/          pathways CSV, notes/ (reading notes)
  notes/        PROJECT_BRIEF.md (this file), DECISIONS.md
  from_IISE/    read-only background (IISE paper, letter, STRESS checklist, old code) - never copied from
  bib/          copies of the two source .bib files (can be deleted; master is in ../ED Reference Library)
../ED Reference Library/ed_library.bib   master bibliography for all ED studies
```

## 12. Open items / next steps

1. Literature review: ED streaming, prioritization rules, policy fit across ED types, and KPI sets.
2. Fix the ED-type factors and their levels; find published sources for each archetype's parameters.
3. Set up the clean package and GitHub repository with the number-integrity pipeline (Section 10a) from the first commit.
4. Specify the patient generator and the pathway sampler (from the Value in Health table).
5. Write the propositions for the heuristic-to-rule equivalence (Section 6).
6. Check TEM author guidelines: length, review type (single or double blind), data and code policy.
