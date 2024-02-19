"""
General testing of benchmark setup.
These test only ensure that the benchmarks can be called with all parameters.
The tests only test the general behaviour but no details.
"""

import pytest
import RnaBench
import numpy as np

from torchvision import transforms

from RnaBench.lib.utils import pairs2db
from RnaBench.lib.rna_design_algorithms.gca import DeterministicGCA
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.feature_extractors import StructuralMotifs
from RnaBench.lib.datasets import ToTensor

@pytest.mark.parametrize("matrix", [True, False])
@pytest.mark.parametrize("task", ['inverse_rna_folding',
                                  'constrained_design',
                                  'inter_family',
                                  'intra_family',
                                  'biophysical_model',
                                  'inter_family_fine_tuning',
                                  'Riboswitch_design'])
@pytest.mark.parametrize("nc", [True, False])
@pytest.mark.parametrize("pks", [True, False])
@pytest.mark.parametrize("multiplets", [True, False])
@pytest.mark.parametrize("min_length", [None, 30])
@pytest.mark.parametrize("max_length", [None, 300])
@pytest.mark.parametrize("feature_extractors", [None, {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")}])
@pytest.mark.parametrize("transform", [True, False])
def test_general_behaviour(matrix,
                           feature_extractors,
                           task,
                           nc,
                           pks,
                           multiplets,
                           min_length,
                           max_length,
                           transform,
                           ):
    # design_model = DeterministicGCA()
    # folding_model = RNAFold()

    # def design_wrapper(rna):
    #     pred = design_model(pairs2db(rna.pairs))
    #     return pred

    # def folding_wrapper(rna):
    #     pred, energy = folding_model(rna.sequence)
    #     return pred

    # def riboswitch_design_wrapper():
    #     rng = np.random.default_rng(seed=0)
    #     preds = []
    #     for i in range(10):
    #         size = rng.integers(66, 92)
    #         seq = rng.choice(['A', 'C', 'G', 'U'], size=size)
    #         preds.append(seq)
    #     return preds

    # bd = RnaBench.RnaDesignBenchmark()
    # bf = RnaBench.RnaFoldingBenchmark()
    # br = RnaBench.RiboswitchDesignBenchmark()

    if transform:
        composed = transforms.Compose([ToTensor(device='cpu')])
    else:
        composed = None

    if task in ['inverse_rna_folding', 'constrained_design']:
        # rna_design mainly has nc pairs, so we do not test design without nc pairs.
        nc = True
        bd = RnaBench.RnaDesignBenchmark(
                                        task=task,
                                        min_length=min_length,
                                        max_length=max_length,
                                        pks=pks,
                                        multiplets=multiplets,
                                        nc=nc,
                                        feature_extractors=feature_extractors,
                                        )

        train, valid, test = bd.get_datasets(
                                            task=task,
                                            min_length=min_length,
                                            max_length=max_length,
                                            pks=pks,
                                            multiplets=multiplets,
                                            nc=nc,
                                            feature_extractors=feature_extractors,
                                            matrix=matrix,
                                            )


        train, valid, test = bd.get_torch_datasets(
                                            task=task,
                                            min_length=min_length,
                                            max_length=max_length,
                                            pks=pks,
                                            multiplets=multiplets,
                                            nc=nc,
                                            feature_extractors=feature_extractors,
                                            matrix=matrix,
                                            transform=composed,
                                            )

        train, valid, test = bd.get_iterators(
                                           device='cpu',
                                           matrix=matrix,
                                           feature_extractors=feature_extractors,
                                           task=task,
                                           nc=nc,
                                           pks=pks,
                                           multiplets=multiplets,
                                           min_length=min_length,
                                           max_length=max_length,
                                           )
        # bd = RnaBench.RnaDesignBenchmark(task=task, max_length=50, timeout=1)
        # bd(design_wrapper, save_results=False)

    elif task in ['inter_family', 'intra_family', 'biophysical_model']:
        bf = RnaBench.RnaFoldingBenchmark(
                                        task=task,
                                        min_length=min_length,
                                        max_length=max_length,
                                        pks=pks,
                                        multiplets=multiplets,
                                        nc=nc,
                                        feature_extractors=feature_extractors,
                                        )

        # bf = RnaBench.RnaFoldingBenchmark(task=task, max_length=50)
        # bf(folding_wrapper, save_results=False)

        train, valid, test = bf.get_datasets(
                                            task=task,
                                            min_length=min_length,
                                            max_length=max_length,
                                            pks=pks,
                                            multiplets=multiplets,
                                            nc=nc,
                                            feature_extractors=feature_extractors,
                                            matrix=matrix,
                                            )



        train, valid, test = bf.get_torch_datasets(
                                          task=task,
                                          min_length=min_length,
                                          max_length=max_length,
                                          pks=pks,
                                          multiplets=multiplets,
                                          nc=nc,
                                          feature_extractors=feature_extractors,
                                          matrix=matrix,
                                          transform=composed,
                                          )

        train, valid, test = bf.get_iterators(
                                           device='cpu',
                                           matrix=matrix,
                                           feature_extractors=feature_extractors,
                                           task=task,
                                           nc=nc,
                                           pks=pks,
                                           multiplets=multiplets,
                                           min_length=min_length,
                                           max_length=max_length,
                                           )

    # br(riboswitch_design_wrapper, save_results=False)
    br = RnaBench.RiboswitchDesignBenchmark()

    train, valid, test = br.get_datasets(
                                        task=task,
                                        min_length=min_length,
                                        max_length=max_length,
                                        pks=pks,
                                        multiplets=multiplets,
                                        nc=nc,
                                        feature_extractors=feature_extractors,
                                        matrix=matrix,
                                        )



    train, valid, test = br.get_torch_datasets(
                                        task=task,
                                        min_length=min_length,
                                        max_length=max_length,
                                        pks=pks,
                                        multiplets=multiplets,
                                        nc=nc,
                                        feature_extractors=feature_extractors,
                                        matrix=matrix,
                                        transform=composed,
                                        )



    train, valid, test = br.get_iterators(
                                       device='cpu',
                                       matrix=matrix,
                                       feature_extractors=feature_extractors,
                                       task=task,
                                       nc=nc,
                                       pks=pks,
                                       multiplets=multiplets,
                                       min_length=min_length,
                                       max_length=max_length,
                                       )


