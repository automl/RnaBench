import RnaBench

from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
#from RnaBench.lib.rna_folding_algorithms.contrafold import ContraFold
#from RnaBench.lib.rna_folding_algorithms.ipknot import IpKnot
#from RnaBench.lib.rna_folding_algorithms.pkiss import PKiss
#from RnaBench.lib.rna_folding_algorithms.linearfold import LinearFoldC, LinearFoldV
from RnaBench.lib.rna_folding_algorithms.rnastructure import Fold
#from RnaBench.lib.rna_folding_algorithms.DL.spotrna import SpotRna
#from RnaBench.lib.rna_folding_algorithms.DL.mxfold2 import MxFold2
from RnaBench.lib.utils import db2pairs

def prediction_wrapper(rna_folding_task):
    # RNAfold requires the sequence in in string format
    sequence = ''.join(rna_folding_task.sequence)
    # RNAfold returns tuple of list of pairs and energy
    pred_pairs = model(sequence)

    return pred_pairs

def rnafold_wrapper(rna_folding_task):
    # RNAfold requires the sequence in in string format
    sequence = ''.join(rna_folding_task.sequence)
    # RNAfold returns tuple of list of pairs and energy
    pred_pairs, energy = model(sequence)

    return pred_pairs


def spotrna_wrapper(rna_folding_task):

    sequence = rna_folding_task.sequence

    pred_pairs = model(sequence)

    return pred_pairs


# instantiate the folding benchmark
folding_benchmark = RnaBench.RnaFoldingBenchmark()

algorithms = [
    RNAFold,
    # ContraFold,
    # IpKnot,
    # PKiss,
    # LinearFoldV,
    # LinearFoldC,
    # # Fold,
    #SpotRna,
    # MxFold2,
]

for algorithm in algorithms:
    # instantiate your model, here exemplarily using RNAfold
    model = algorithm()
    algorithm_name = model.__name__()
    print('### Start evaluation of', algorithm_name)

    if algorithm_name == 'RNAFold':
        metrics = folding_benchmark(rnafold_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )
    elif algorithm_name == 'SPOT-RNA':
        metrics = folding_benchmark(spotrna_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )
    else:
        metrics = folding_benchmark(prediction_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )

    # RnaBench will compute several metrics for your model predictions
    print(f"{algorithm_name} results:")
    print(metrics)

