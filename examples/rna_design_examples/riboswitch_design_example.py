"""
In this example, we show the design of Riboswitches using the original procedure
proposed by Wachsmuth et al. 2012.
"""

import RnaBench

from RnaBench.lib.rna_design_algorithms.riboswitch_design import OriginalProcedure
from RnaBench.lib.feature_extractors import StructuralMotifs

# the feature extractors are used to get sequence or structure features for evaluations
# with the distribution learning metrics like novelty or diversity.
feature_extractors = {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists"),
                      }
# Instantiate the benchmark
riboswitch_benchmark = RnaBench.RiboswitchDesignBenchmark(feature_extractors=feature_extractors)

# and the model. Here, we predcit 1000 candidates
model = OriginalProcedure(n_candidates=1000)

# define the prediction wrapper function
def riboswitch_design_wrapper(*args, **kwargs):
    predictions = model()
    return predictions

# and get the metrics.
metrics = riboswitch_benchmark(riboswitch_design_wrapper)

print(metrics)


