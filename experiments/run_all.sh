#!/usr/bin/env bash
# Runs every experiment from the committed model, then builds all outputs.
set -euo pipefail
cd "$(dirname "$0")/scripts"
unset EDSIM_QUICK
for s in e6_optimization e7_transfer e6b_levers e8_robustness e9_out_of_sample e0_pilot e1_validation e3_fidelity e2_main e5_sensitivity e4_alpha; do
  echo "=== $s $(date -u +%H:%M:%S)"
  python3 "$s.py"
done
echo "=== outputs $(date -u +%H:%M:%S)"
python3 make_outputs.py
echo "=== done $(date -u +%H:%M:%S)"
