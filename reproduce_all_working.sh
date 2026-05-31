#!/usr/bin/env bash
# Working version of reproduce_all.sh after the 2026-05-30 audit.
#
# Notes vs. the original reproduce_all.sh:
#   * The original sources its baselines from `examples/evaluate_all_folding_baselines_example.py`,
#     which in the public commit has every classical baseline commented out and only
#     RNAformer + UFold uncommented. This script preserves that — see §3.
#   * MPLBACKEND=Agg + plt.show() suppression keep the matplotlib viz steps non-interactive.
#   * PATH is extended to find the locally-built LinearFold, IpKnot, the VARNA wrapper,
#     and the RnaBench conda env's binaries (mxfold2, contrafold, pKiss, Fold, ct2dot, …).
#   * DATAPATH is set to the conda env's RNAstructure energy parameter tables.
#
# Prerequisites (one-time, see STATUS.md §4):
#   * RnaBench conda env populated per STATUS.md §4
#   * `python -m RnaBench.download` already run (~300 MB)
#   * data/3D.tar.gz and RnaBench/lib/data/CMs.tar.gz extracted
#   * external_algorithms/ contains:
#       - LinearFold/  (git clone + make)
#       - ipknot       (gdown 1Oh3kNYbnv_22i4IIYXavPcOo1xSdm1vB)
#       - SPOT-RNA/    (git clone + Dropbox SPOT-RNA-models.tar.gz extracted in-place)
#       - VARNA/VARNAv3-93.jar + ./varna shell wrapper

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${REPO_DIR}"

PY="${PY:-/home/dominik/.conda/envs/RnaBench/bin/python}"
PY_BIN_DIR="$(dirname "${PY}")"
export PATH="${REPO_DIR}/external_algorithms/VARNA:${REPO_DIR}/external_algorithms/LinearFold:${REPO_DIR}/external_algorithms:${PY_BIN_DIR}:${PATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
RNASTRUCTURE_DATA="$(dirname "${PY_BIN_DIR}")/share/rnastructure/data_tables"
if [ -d "${RNASTRUCTURE_DATA}" ]; then
    export DATAPATH="${RNASTRUCTURE_DATA}"
fi

echo "### Run all folding predictions (RNAformer + UFold via examples.evaluate_all_folding_baselines_example)"
"${PY}" -m examples.evaluate_all_folding_baselines_example

echo "### Visualize folding predictions"
"${PY}" -m examples.visualize_folding_predictions

echo "### Run inverse RNA folding benchmark (GCA + all classical folding baselines)"
"${PY}" -m examples.rna_design_examples.inverse_rna_folding_example

echo "### Plot Inverse RNA Folding results"
"${PY}" -m examples.visualize_inverse_rna_folding_predictions

echo "### Visualize 5SrRNA F1/WL/MCC comparison"
"${PY}" -m examples.f1score_wl_example

echo "### reproduce_all complete"
