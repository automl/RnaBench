import RnaBench

from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.rnaformer import RNAformer
from RnaBench.lib.utils import db2pairs

# instantiate the folding benchmark
folding_benchmark = RnaBench.RnaFoldingBenchmark(task='inter_family')

# instantiate your model, here exemplarily using RNAfold
model = RNAformer(dim=256, cycling=True)

def prediction_wrapper(rna_folding_task):
    pred_pairs = model(rna_folding_task.sequence)

    return pred_pairs

# RnaBench will compute several metrics for your model predictions
metrics = folding_benchmark(prediction_wrapper, save_results=True, algorithm_name=model.__name__())

print(metrics)

