#!/usr/bin/env bash
# Working version of reproduce_minimal.sh after the 2026-05-30 audit.
#
# Differences vs. the original reproduce_minimal.sh:
#   * Sets MPLBACKEND=Agg so the matplotlib plt.show() calls don't block.
#   * Prepends the local VARNA wrapper to PATH so visualize_rna() works.
#   * Uses the RnaBench conda env python directly (no need to `conda activate`).
#   * Aborts on the first failing step (set -e).
#
# Prerequisites:
#   * `conda env create -f environment.yml` (or the recipe in STATUS.md §4)
#   * `python -m RnaBench.download` has been run at least once (~300 MB)
#   * The 3D and CMs tarballs have been extracted:
#       tar -xzf data/3D.tar.gz -C data
#       tar -xzf RnaBench/lib/data/CMs.tar.gz -C RnaBench/lib/data
#   * VARNA jar present at external_algorithms/VARNA/VARNAv3-93.jar with
#     the matching `varna` shell wrapper.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${REPO_DIR}"

PY="${PY:-/home/dominik/.conda/envs/RnaBench/bin/python}"
PY_BIN_DIR="$(dirname "${PY}")"
export PATH="${REPO_DIR}/external_algorithms/VARNA:${REPO_DIR}/external_algorithms/LinearFold:${REPO_DIR}/external_algorithms:${PY_BIN_DIR}:${PATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
# Point RNAstructure's `Fold`/`ct2dot` at the energy parameter tables
# installed by the bioconda `rnastructure` package.
RNASTRUCTURE_DATA="$(dirname "${PY_BIN_DIR}")/share/rnastructure/data_tables"
if [ -d "${RNASTRUCTURE_DATA}" ]; then
    export DATAPATH="${RNASTRUCTURE_DATA}"
fi

echo "### Folding baselines (minimal — RNAFold only)"
"${PY}" -m examples.evaluate_all_folding_baselines_example_minimal

echo "### Visualize folding predictions"
"${PY}" -m examples.visualize_folding_predictions

echo "### Inverse RNA folding benchmark (minimal — GCA)"
"${PY}" -m examples.rna_design_examples.inverse_rna_folding_example_minimal

echo "### Plot inverse RNA folding results"
"${PY}" -m examples.visualize_inverse_rna_folding_predictions

echo "### Compare F1/WL/MCC on shifted 5S rRNA"
"${PY}" -m examples.f1score_wl_example

echo "### Plot dataset length distributions"
"${PY}" -m examples.visualize_data_length_dist

echo "### Plot per-sample structure representations"
"${PY}" -m examples.visualize_representations

echo "### reproduce_minimal complete"
