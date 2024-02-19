"""
This is an example for the constrained design benchmark.

We use the DeterministicGCA algorithm as the model for predicting RNA sequences
that fold into a desired secondary structure.
For the constrained design, additional sequence constraints are provided with
the sequence.

The folding algorithm that we use is RNAFold.
"""

import RnaBench

from RnaBench.lib.rna_design_algorithms.rnainverse import RNAInverse
from RnaBench.lib.rna_design_algorithms.gca import DeterministicGCA
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import pairs2db


# instantiate your model
model = DeterministicGCA()

# instantiate the benchmark
design_benchmark = RnaBench.RnaDesignBenchmark(task='constrained_design',
                                               timeout=1)

def prediction_wrapper(rna_design_task, *args, **kwargs):
    structure = pairs2db(rna_design_task.pairs, rna_design_task.sequence)

    # prediction also considers sequence constraints
    sequence = model(structure, rna_design_task.sequence)

    sequence = list(sequence)

    return sequence

# you can do multiple evaluation runs
num_iterations = 1
for i in range(num_iterations):
    # Given a folding algorithm, RnaBench computes several metrics
    metrics = design_benchmark(prediction_wrapper, folding_algorithm=RNAFold(), algorithm_name='DeterministicGCA')

    print(metrics)
