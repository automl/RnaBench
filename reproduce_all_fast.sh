#!/usr/bin/env bash
# Time-bounded subset variant of reproduce_all_working.sh.
#
# Runs every baseline that reproduce_all touches (RNAformer dim64, UFold,
# every classical folding baseline used by inverse_rna_folding_example) but
# limits each benchmark to max_length=80 / first ~20 samples so the whole
# pipeline completes in a few minutes instead of many hours. Useful as a
# smoke / CI run; for paper-grade numbers use reproduce_all_working.sh.

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

echo "### Folding (subset): RNAformer dim64 + UFold ufold_train.pt nc=False"
"${PY}" -c "
import sys, os
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
import RnaBench
from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.rnaformer import RNAformer
from RnaBench.lib.rna_folding_algorithms.DL.ufold.ufold import UFold

bench = RnaBench.RnaFoldingBenchmark(task='intra_family', feature_extractors=None, max_length=80)
print('# samples:', len(bench.data))

m1 = RNAformer(dim=64, cycling=False)
def w1(rna):
    return m1(rna.sequence)
print('RNAformer dim64:', bench(w1, save_results=True, algorithm_name=m1.__name__()))

m2 = UFold(model='ufold_train.pt', nc=False)
def w2(rna):
    return m2(rna.sequence)
print('UFold:', bench(w2, save_results=True, algorithm_name=m2.__name__()))
"

echo "### Visualize folding predictions"
"${PY}" -m examples.visualize_folding_predictions

echo "### Inverse RNA folding (subset, GCA + all classical baselines)"
"${PY}" -c "
import sys, os
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
import RnaBench
from RnaBench.lib.rna_design_algorithms.gca import DeterministicGCA
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.rna_folding_algorithms.contrafold import ContraFold
from RnaBench.lib.rna_folding_algorithms.ipknot import IpKnot
from RnaBench.lib.rna_folding_algorithms.pkiss import PKiss
from RnaBench.lib.rna_folding_algorithms.linearfold import LinearFoldC, LinearFoldV
from RnaBench.lib.rna_folding_algorithms.rnastructure import Fold
from RnaBench.lib.rna_folding_algorithms.DL.spotrna import SpotRna
from RnaBench.lib.rna_folding_algorithms.DL.mxfold2 import MxFold2
from RnaBench.lib.utils import pairs2db
from RnaBench.lib.feature_extractors import StructuralMotifs

algorithms = [RNAFold, ContraFold, IpKnot, PKiss, LinearFoldV, LinearFoldC, Fold, SpotRna, MxFold2]

model = DeterministicGCA()
feature_extractors = {'structural_motifs': StructuralMotifs(source='forgi', aggregation_level='motif_lists')}
b = RnaBench.RnaDesignBenchmark(
    task='inverse_rna_folding', timeout=5, pks=False, multiplets=False,
    feature_extractors=feature_extractors, max_length=80,
)

def w(rna):
    return list(model(pairs2db(rna.pairs, rna.sequence)))

for algo in algorithms:
    f = algo()
    print(f'### Evaluation with {f.__name__()}')
    m = b(w, folding_algorithm=f, algorithm_name='DeterministicGCA')
    print(m)
"

echo "### Plot inverse RNA folding results"
"${PY}" -m examples.visualize_inverse_rna_folding_predictions

echo "### Compare F1/WL/MCC on shifted 5S rRNA"
"${PY}" -m examples.f1score_wl_example

echo "### reproduce_all_fast complete"
