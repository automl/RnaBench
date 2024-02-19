"""
In this example we show how to add properties to the Riboswitch design benchmark.
"""

import RnaBench

from RnaBench.lib.rna_design_algorithms.riboswitch_design import OriginalProcedure

riboswitch_benchmark = RnaBench.RiboswitchDesignBenchmark()

model = OriginalProcedure(n_candidates=1000)

def riboswitch_design_wrapper(*args, **kwargs):
    predictions = model()
    return predictions

metrics = riboswitch_benchmark(riboswitch_design_wrapper, desired_gc=0.5, desired_energy=-26.0)

print(metrics)
