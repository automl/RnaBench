import RnaBench

from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.rna_folding_algorithms.contrafold import ContraFold
from RnaBench.lib.rna_folding_algorithms.ipknot import IpKnot
from RnaBench.lib.rna_folding_algorithms.pkiss import PKiss
from RnaBench.lib.rna_folding_algorithms.linearfold import LinearFoldC, LinearFoldV
from RnaBench.lib.rna_folding_algorithms.rnastructure import Fold
from RnaBench.lib.rna_folding_algorithms.DL.spotrna import SpotRna
from RnaBench.lib.rna_folding_algorithms.DL.mxfold2 import MxFold2
from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.rnaformer import RNAformer
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.probtransformer import ProbabilisticTransformer
from RnaBench.lib.rna_folding_algorithms.DL.ufold.ufold import UFold
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

def ProbTransformer_wrapper(rna_folding_task):

    sequence = rna_folding_task.sequence

    pred_pairs = model(sequence)

    return pred_pairs


def RNAformer_wrapper(rna_folding_task):

    sequence = rna_folding_task.sequence

    pred_pairs = model(sequence)

    return pred_pairs

def ufold_wrapper(rna_folding_task):

    sequence = rna_folding_task.sequence

    pred_pairs = model(sequence)

    return pred_pairs

# instantiate the folding benchmark
folding_benchmark = RnaBench.RnaFoldingBenchmark(task='intra_family', feature_extractors=None)

algorithms = [
    # RNAFold,
    # ContraFold,
    # IpKnot,
    # PKiss,
    # LinearFoldV,
    # LinearFoldC,
    # # Fold,
    # SpotRna,
    # MxFold2,
    # ProbabilisticTransformer,
    RNAformer,
    UFold,
]

for algorithm in algorithms:
    # instantiate your model, here exemplarily using RNAfold
    # print('### Initiate model')
    model = algorithm()
    algorithm_name = model.__name__()

    print('### Initialize evaluation of', algorithm_name)
    
    if algorithm_name == 'RNAFold':
        model = algorithm()  # we initialize the model twice here because we select via algorithm name which is suboptimal of course.
        algorithm_name = model.__name__()
        print('### Start evaluation of', algorithm_name)

        metrics = folding_benchmark(rnafold_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )
        print(f"{algorithm_name} results:")
        print(metrics)
    elif algorithm_name == 'SPOT-RNA':
        model = algorithm()
        algorithm_name = model.__name__()
        print('### Start evaluation of', algorithm_name)

        metrics = folding_benchmark(spotrna_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )
        print(f"{algorithm_name} results:")
        print(metrics)    
    elif 'RNAformer' in algorithm_name:
        for d in [64, 128, 256]:  # 64, 
            for cyc in [True, False]:
                if d != 256:
                    if cyc:
                        continue
                model = algorithm(dim=d, cycling=cyc)
                algorithm_name = model.__name__()
                print('### Start evaluation of', algorithm_name)

                metrics = folding_benchmark(RNAformer_wrapper,
                                            save_results=True,
                                            algorithm_name=algorithm_name,
                                            )
                print(f"{algorithm_name} results:")
                print(metrics)
    elif 'ProbTransformer' in algorithm_name:
        model = algorithm()
        algorithm_name = model.__name__()
        print('### Start evaluation of', algorithm_name)

        metrics = folding_benchmark(ProbTransformer_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )
        print(f"{algorithm_name} results:")
        print(metrics)

    elif 'Ufold' in algorithm_name:
        for m in ['ufold_train.pt', 'ufold_train_alldata.pt', 'ufold_train_pdbfinetune.pt']:
            for nc in [True, False]:
                model = algorithm(model=m, nc=nc)
                algorithm_name = model.__name__()
                print('### Start evaluation of', algorithm_name)
    
                metrics = folding_benchmark(ufold_wrapper,
                                            save_results=True,
                                            algorithm_name=algorithm_name,
                                            )
                print(f"{algorithm_name} results:")
                print(metrics)
    else:
        model = algorithm()
        algorithm_name = model.__name__()
        print('### Start evaluation of', algorithm_name)
        metrics = folding_benchmark(prediction_wrapper,
                                    save_results=True,
                                    algorithm_name=algorithm_name,
                                    )
        print(f"{algorithm_name} results:")
        print(metrics)


    # # RnaBench will compute several metrics for your model predictions
    # print(f"{algorithm_name} results:")
    # print(metrics)

