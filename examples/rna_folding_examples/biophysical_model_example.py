import RnaBench

from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import db2pairs

# instantiate the folding benchmark
folding_benchmark = RnaBench.RnaFoldingBenchmark(task='biophysical_model')

# instantiate your model, here exemplarily using RNAfold
model = RNAFold()

def prediction_wrapper(rna_folding_task):
    # RNAfold requires the sequence in in string format
    # sequence = ''.join(rna_folding_task.sequence)
    # RNAfold returns tuple of list of pairs and energy
    pred_pairs, energy = model(rna_folding_task.sequence)

    return pred_pairs

# RnaBench will compute several metrics for your model predictions
metrics = folding_benchmark(prediction_wrapper, save_results=True, algorithm_name='RNAFold')

print(metrics)
