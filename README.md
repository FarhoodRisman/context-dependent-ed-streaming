# One Policy Does Not Fit All

*Matching Patient Streaming Policies to Emergency Department Types* (target: IEEE Transactions on Engineering Management)

A discrete-event simulation study of emergency department (ED) streaming policies (queue rule plus routing rule) across 27 synthetic ED types, with a regret analysis of one-size-fits-all policies, a prediction-accuracy sweep, and a fidelity analysis of similarity-based scoring rules.

Start with `notes/PROJECT_BRIEF.md`; the model is specified in `notes/MODEL_SPEC.md`; decisions are logged in `notes/DECISIONS.md`.

## Repository layout

| Folder | Contents |
|---|---|
| `sim/` | Python package `edsim` (`src/edsim`), tests (`tests/`), configuration files (`configs/`): every model input with its source |
| `experiments/` | Experiment scripts (`scripts/e0_pilot.py` … `e5_sensitivity.py`), result checks, the output builder, and raw results with provenance (`results/`) |
| `paper/` | LaTeX manuscript: `main.tex` (two-column, with authors), `main_review.tex` (anonymised, double-spaced, for submission), `sections/`, generated `tables/`, `figures/`, `numbers.tex`, and `supplement/` |
| `lit/` | Pathway table from Rismanchian et al. (2023, *Value in Health*) |
| `notes/` | Project brief, model specification, decisions log, workflow |

## Reproduce everything

```bash
pip install -e sim[test]          # numpy, pandas, scipy, matplotlib, scikit-learn, pytest
cd sim && pytest -q && cd ..      # verification tests
experiments/run_all.sh            # E0-E5 (about 1.5 h on two cores), checks, then all outputs
cd paper && latexmk -pdf main.tex main_review.tex
```

`experiments/scripts/make_outputs.py` runs `check_results.py` first and refuses to build outputs if any check fails. It writes every table (`paper/tables/*.tex`), figure (`paper/figures/*.pdf`) and quoted number (`paper/numbers.tex`, LaTeX macros), plus `paper/outputs_provenance.json` recording the code commit, configuration hash and run settings of each results file.

## Number integrity

No result is typed into the manuscript by hand. The results text uses only the macros in `numbers.tex`; tables and figures are generated. To change anything, change the configuration or code, re-run, and rebuild.

## Licence

MIT (see `LICENSE`).
