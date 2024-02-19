"""
Minimal example for the inverse RNA Folding benchmark using only RNAFold as the
folding oracle.

We use RNAInverse as design baseline.
"""

import RnaBench

from RnaBench.lib.rna_design_algorithms.gca import DeterministicGCA
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import pairs2db

algorithms = [
    RNAFold,
    # ContraFold,
    # IpKnot,
    # PKiss,
    # LinearFoldV,
    # LinearFoldC,
    # # Fold,
    # SpotRna,
    # MxFold2,
]


# instantiate your model, here exemplarily using RNAinverse
# model = RNAInverse()
model = DeterministicGCA()

# instantiate the GoalDirected RNA design benchmark
design_benchmark = RnaBench.RnaDesignBenchmark(task='inverse_rna_folding',
                                               timeout=1,
                                               max_length=50,)


def prediction_wrapper(rna_design_task, *args, **kwargs):
    # RNAinverse requires the strutcure in dot-bracket format
    structure = pairs2db(rna_design_task.pairs, rna_design_task.sequence)
    # RNAInverse predicts the sequence;
    # returns ['A'] * len(structure) if no solution found
    sequence = model(structure)
    sequence = list(sequence)

    return sequence

for algorithm in algorithms:
    print('### Evaluation with', algorithm().__name__())
    metrics = design_benchmark(prediction_wrapper, folding_algorithm=algorithm(), algorithm_name='DeterministicGCA')
    print(metrics)
