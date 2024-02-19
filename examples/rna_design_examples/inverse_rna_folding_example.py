"""
In this example, the DeterministicGCA baseline for inverse RNA folding
is evaluated with all folding baselines on tasks without pseudoknots.

Running this script takes a while.
"""

import RnaBench

from RnaBench.lib.rna_design_algorithms.rnainverse import RNAInverse
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

algorithms = [
    RNAFold,
    ContraFold,
    IpKnot,
    PKiss,
    LinearFoldV,
    LinearFoldC,
    SpotRna,
    MxFold2,
]


# instantiate your model, here exemplarily using RNAinverse
# model = RNAInverse()
model = DeterministicGCA()

# Feature extractors can extract certain features from the sequence or the structures
# these can be used to compute the distribution learning metrics like diverstiy, novelty, and KL divergence.
feature_extractors = {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists"),
                      }
# instantiate the GoalDirected RNA design benchmark
design_benchmark = RnaBench.RnaDesignBenchmark(task='inverse_rna_folding',
                                               timeout=5,
                                               pks=False,
                                               multiplets=False,
                                               feature_extractors=feature_extractors,)


def prediction_wrapper(rna_design_task, *args, **kwargs):
    structure = pairs2db(rna_design_task.pairs, rna_design_task.sequence)
    sequence = model(structure)
    sequence = list(sequence)

    return sequence

for algorithm in algorithms:
    print('### Evaluation with', algorithm().__name__())
    metrics = design_benchmark(prediction_wrapper, folding_algorithm=algorithm(), algorithm_name='DeterministicGCA')
    print(metrics)
