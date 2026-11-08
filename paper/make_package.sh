#!/usr/bin/env bash
# Builds the PDFs and the LaTeX submission package (zip) from the generated outputs.
# Local builds use natbib's unsrtnat when IEEEtranN.bst is not installed; Overleaf and
# TeX Live installations with the IEEEtran bundle use IEEEtranN automatically.
set -euo pipefail
cd "$(dirname "$0")"
PRETEX=""
if ! kpsewhich IEEEtranN.bst > /dev/null; then PRETEX='\def\localbst{1}'; fi
for f in main main_review; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error ${PRETEX:+-usepretex="$PRETEX"} "$f.tex" > /dev/null
done
(cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error STRESS_DES_checklist.tex > /dev/null)
# number audit: no digits typed in the results text outside macros, tables and figures
python3 ../experiments/scripts/audit_numbers.py
STAMP=$(date -u +%Y%m%d)
OUT=OPDNFA_TEM_LaTeX_${STAMP}.zip
rm -f "$OUT"
zip -q -r "$OUT" main.tex main_review.tex preamble.tex numbers.tex IEEEtran.cls ed_library.bib \
  sections tables figures outputs_provenance.json main.pdf main_review.pdf \
  supplement/STRESS_DES_checklist.tex supplement/STRESS_DES_checklist.pdf supplement/*.csv supplement/*.txt \
  README_package.txt
echo "$OUT"
