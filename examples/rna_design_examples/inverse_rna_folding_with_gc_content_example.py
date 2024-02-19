"""
In this example, we show how one can add a desired GC content to the inverse
RNA Folding benchmark.

We use the DeterministicGCA as design baseline with a very short timeout of 1 second.
"""

import RnaBench

from RnaBench.lib.rna_design_algorithms.rnainverse import RNAInverse
from RnaBench.lib.rna_design_algorithms.gca import DeterministicGCA
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import pairs2db


# instantiate your model, here exemplarily using RNAinverse
model = DeterministicGCA()

# instantiate the benchmark
design_benchmark = RnaBench.RnaDesignBenchmark(task='inverse_rna_folding',
                                               timeout=1)


def prediction_wrapper(rna_design_task, *args, **kwargs):
    structure = pairs2db(rna_design_task.pairs, rna_design_task.sequence)
    sequence = model(structure)
    sequence = list(sequence)

    return sequence

# you can do multiple evaluation runs
num_iterations = 1
for i in range(num_iterations):
    # Given a folding algorithm, RnaBench computes several metrics
    metrics = design_benchmark(prediction_wrapper, folding_algorithm=RNAFold(), desired_gc=0.5, algorithm_name='DeterministicGCA')

    print(metrics)
