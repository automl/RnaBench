import RnaBench

import pickle
import pandas as pd
import numpy as np

from enum import Enum, unique
from typing import List, Optional, Dict, Callable
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms

from RnaBench.download import select_and_download
from RnaBench.lib.datasets import RnaDataset, TorchDataset, ToTensor
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.rna_folding_algorithms.rnastructure import Fold
from RnaBench.lib.execution import timing, limited_execution
from RnaBench.lib.metrics import GoalDirectedMetrics, RiboswitchDesignMetrics, DistributionLearningMetrics, DistributionLearningComparativeMetrics
from RnaBench.lib.alphabets import Nucleotide, Structure
from RnaBench.lib.feature_extractors import StructuralMotifs


@unique
class Tasks(Enum):
    inverse_rna_folding = "inverse_rna_folding"
    constrained_design = 'constrained_design'
    RiboswitchDesign = 'riboswitch_design'
    intraFamily = 'intra_family'
    interFamily = 'inter_family'
    biophysical_model = "biophysical_model"
    inter_family_fine_tuning = "inter_family_fine_tuning"


class Benchmark():
    def __init__(self, **kwargs):
        for k, arg in kwargs.items():
            setattr(self, k, arg)

    def load_data(self, save_new_features=True):
        if self.task.value == 'inter_family_fine_tuning':
            data_path = Path(self.data_dir, 'inter_family_benchmark.plk.gz')
        else:
            data_path = Path(self.data_dir, self.task.value + '_benchmark.plk.gz')

        if not data_path.is_file():
            Path(self.data_dir).mkdir(exist_ok=True, parents=True)
            select_and_download(self.task.value,
                                save_dir=self.data_dir,
                                )

        self.data = pd.read_pickle(data_path)
        if self.min_length is not None:
            self.data = self.data[self.data['sequence'].apply(lambda x: self.min_length <= len(x))]
        if self.max_length is not None:
            self.data = self.data[self.data['sequence'].apply(lambda x: len(x) <= self.max_length)]

        if not self.nc:
            self.data = self.data[self.data['has_nc'] == False]

        if not self.pks:
            self.data = self.data[self.data['has_pk'] == False]

        if not self.multiplets:
            self.data = self.data[self.data['has_multiplet'] == False]


        if self.feature_extractors is not None:
            self.data = self.extract_features(self.data)

#             if save_new_features:
#                 now = datetime.now()
#                 now = now.strftime("%d/%m/%Y %H:%M:%S")
#
#                 Path(self.results_dir, 'RNA_design_with_features', self.algorithm_name).mkdir(exist_ok=True, parents=True)
#
#                 out_path = Path(self.results_dir, 'RNA_design_with_features', self.algorithm_name,
#                                 f"{'_'.join([str(x) for x in [self.algorithm_name, self._evaluation_counter, 'min_len', self.min_length, 'max_len', self.max_length, 'nc', self.nc, 'pks', self.pks, 'multiplets', self.multiplets, 'timeout', self.timeout]])}_{'-'.join(now.split('/')).replace(' ', '-')}.plk")
#                 with open(out_path, 'wb') as f:
#                     pickle.dump(self.data, f)

    def get_datasets(self,
                     matrix=False,
                     sequence_vocab=Nucleotide.iupac_alphabet,
                     structure_vocab=Structure.extended_dot_bracket,
                     feature_extractors=None,  # {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")},
                     task=None,
                     nc=None,
                     pks=None,
                     multiplets=None,
                     min_length=None,
                     max_length=None,
                    ):
        if task is not None:
            try:
                task = Tasks(task.lower())
            except ValueError as e:
                valid_tasks = [x.value for x in Tasks]
                raise ValueError(f"Unknown value of parameter 'task': '{task}'."
                                     f"Please use one of {valid_tasks}.") from e
        else:
            task = self.task

        if nc is None:
            nc = self.nc
        if pks is None:
            pks = self.pks
        if multiplets is None:
            multiplets = self.multiplets
        if min_length is None:
            min_length = self.min_length
        if max_length is None:
            max_length = self.max_length


        if task.value == 'inter_family_fine_tuning':
            train_path = Path(self.data_dir, task.value + '_train.plk.gz')
            valid_path = Path(self.data_dir, 'inter_family_valid.plk.gz')
            test_path = Path(self.data_dir, 'inter_family_benchmark.plk.gz')
        elif task.value == 'riboswitch_design':
            train_path = Path(self.data_dir, task.value + '_train.plk.gz')
            valid_path = None
            test_path = None
        else:
            train_path = Path(self.data_dir, task.value + '_train.plk.gz')
            valid_path = Path(self.data_dir, task.value + '_valid.plk.gz')
            test_path = Path(self.data_dir, task.value + '_benchmark.plk.gz')

        train = RnaDataset(
          dataset=train_path,
          nc=nc,
          pks=pks,
          multiplets=multiplets,
          min_length=min_length,
          max_length=max_length,
          sequence_vocab=sequence_vocab,
          structure_vocab=structure_vocab,
          feature_extractors=feature_extractors,
          matrix=matrix,
        )
        if valid_path is not None:
            valid = RnaDataset(
              dataset=valid_path,
              nc=nc,
              pks=pks,
              multiplets=multiplets,
              min_length=min_length,
              max_length=max_length,
              sequence_vocab=sequence_vocab,
              structure_vocab=structure_vocab,
              feature_extractors=feature_extractors,
              matrix=matrix,
            )
        else:
            valid = None
        if test_path is not None:
            test = RnaDataset(
              dataset=test_path,
              nc=nc,
              pks=pks,
              multiplets=multiplets,
              min_length=min_length,
              max_length=max_length,
              sequence_vocab=sequence_vocab,
              structure_vocab=structure_vocab,
              feature_extractors=feature_extractors,
              matrix=matrix,
            )
        else:
            test = None

        return train, valid, test


    def get_torch_datasets(self,
                           transform=None,
                           matrix=False,
                           seed=0,
                           sequence_vocab=Nucleotide.iupac_alphabet,
                           structure_vocab=Structure.extended_dot_bracket,
                           feature_extractors=None,  # {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")},
                           task=None,
                           nc=None,
                           pks=None,
                           multiplets=None,
                           min_length=None,
                           max_length=None,
                           ):

        if task is not None:
            try:
                task = Tasks(task.lower())
            except ValueError as e:
                valid_tasks = [x.value for x in Tasks]
                raise ValueError(f"Unknown value of parameter 'task': '{task}'."
                                     f"Please use one of {valid_tasks}.") from e
        else:
            task = self.task

        train, valid, test = self.get_datasets(
                                matrix=matrix,
                                sequence_vocab=sequence_vocab,
                                structure_vocab=structure_vocab,
                                feature_extractors=feature_extractors,
                                task=task.value,
                                nc=nc,
                                pks=pks,
                                multiplets=multiplets,
                                min_length=min_length,
                                max_length=max_length,
                                               )
        return (TorchDataset(d,
                             task=task.value,
                             matrix=matrix,
                             seed=seed,
                             transform=transform
                             )
                if d is not None else None
                for d in [train, valid, test])

    def get_iterators(self,
                      device='cpu',
                      matrix=False,
                      seed=0,
                      sequence_vocab=Nucleotide.iupac_alphabet,
                      structure_vocab=Structure.extended_dot_bracket,
                      feature_extractors=None,  # {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")},
                      batch_size=8,
                      num_workers=0,
                      task=None,
                      nc=None,
                      pks=None,
                      multiplets=None,
                      min_length=None,
                      max_length=None,
                      **kwargs,
                      ):

        transform = transforms.Compose([ToTensor(device=device)])

        train, valid, test = self.get_torch_datasets(
                                transform=[transform],
                                matrix=matrix,
                                seed=seed,
                                sequence_vocab=sequence_vocab,
                                structure_vocab=structure_vocab,
                                feature_extractors=feature_extractors,
                                task=task,
                                nc=nc,
                                pks=pks,
                                multiplets=multiplets,
                                min_length=min_length,
                                max_length=max_length,
                                                    )

        return (DataLoader(d,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           **kwargs,
                           ) if d is not None else None
                           for d in [train, valid, test]
                           )

    def extract_features(self, data):

        for feat, feat_ext in self.feature_extractors.items():
            feat_cols = feat_ext.feat_names if hasattr(feat_ext, "feat_names") else [feat]
            if feat not in data:
                feat_extracted = data.apply(lambda x: feat_ext(x), axis=1, result_type="expand")
                if len(feat_cols) == feat_extracted.shape[1]:
                    feat_extracted.columns = feat_cols
                else:
                    raise UserWarning(f"Data is empty. Please change the data settings.")
                                       # '\n' + 'For:' + '\n' + f"{feat_extracted}" +
                                       # '\n' + 'and' + '\n' + f"{feat_cols}")

                data.loc[:, feat_cols] = feat_extracted.to_numpy()

        return data


class RnaDesignBenchmark(Benchmark):
    _metrics = [
                 'f1_score',
                 'mcc',
                 'wl',
                 'recall',
                 'precision',
                 'specificity',
                 'solved',
                 'f1_shifted',
                 # 'hamming_sequence',
                 ]
    __nov_metrics: List[str] = ["iou", "novelty"]
    __div_metrics: List[str] = ["diversity", "diameter", "DPP", "sum_btlnk"]
    __gen_dists: List[str] = ["hamming", ] # "lv"]
    __feats: List[str] = ["sequence", "structure"]
    __feat_distmap: Dict[str, List[str]] = {"predicted_sequence": ["hamming", ],  #"lv"],
                                            "predicted_s_0" : ["l2"],
                                            "predicted_h_0" : ["l2"],
                                            "predicted_i_0" : ["l2"],
                                            "predicted_i_1": ["l2"],
                                            # "predicted_pairs": ["wl_pairs"],
                                            }
                                            #"structure": ["hamming"],
                                            # "predicted_pairs": ["wl_pairs"]}

    _evaluation_counter = 0

    def __init__(
                 self,
                 task : str = 'inverse_rna_folding',
                 nc: bool = True,
                 pks: bool = True,
                 multiplets: bool = False,
                 data_dir: str = 'data',
                 timeout: int = 600,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 results_dir: str = 'results',
                 feature_extractors: Optional[Dict[str, Callable]] = {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")},
                 ):

        # get task
        try:
            self.task = Tasks(task.lower())
        except ValueError as e:
            valid_tasks = [x.value for x in Tasks]
            raise ValueError(f"Unknown value of parameter 'task': '{task}'."
                                 f"Please use one of {valid_tasks}.") from e

        # if 'gc' in self.task.value:
        #     self._metrics = self._metrics + ['gc_content_absolute']

        self.gd_metrics = GoalDirectedMetrics(self._metrics)
        # nov_metrics: List[str] = None,
        #          div_metrics: List[str] = None,
        #          feats=None,
        #          feat_spec_dists=None,
        self.dl_metrics = DistributionLearningMetrics(general_dists=self.__gen_dists, nov_metrics=self.__nov_metrics, div_metrics=self.__div_metrics, feats=self.__feats, feat_spec_dists=self.__feat_distmap)
        self.feature_extractors = feature_extractors
        super().__init__(
                         timeout=timeout,
                         min_length=min_length,
                         max_length=max_length,
                         nc=nc,
                         pks=pks,
                         multiplets=multiplets,
                         results_dir=results_dir,
                         data_dir=data_dir,
                        )
        self.load_data()

    def __call__(
                 self,
                 wrapper_function,
                 *args,
                 folding_algorithm=None,
                 save_results=True,
                 gc_tolerance=0.01,
                 desired_gc=None,
                 dataset=None,
                 task_ids=None,
                 results_path=None,
                 algorithm_name: str = 'custom',
                 **kwargs,
                 ):
        self._evaluation_counter += 1
        if dataset is not None:
            if isinstance(dataset, RnaDataset):
                self._results = dataset.data.copy()
            elif isinstance(dataset, pd.DataFrame):
                self._results = dataset.copy()
            else:
                raise UserWarning(f"Dataset type {type(dataset)} currently not supported")

            if self.feature_extractors is not None:
                # check if features are already extracted
                self._results = self.extract_features(self._results)

        else:
            self._results = self.data.copy()

        if task_ids is not None:
            if isinstance(task_ids, int):
                self._results = self._results[self._results['Id'].isin([task_ids])]
            elif isinstance(task_ids, list):
                self._results = self._results[self._results['Id'].isin(task_ids)]
            elif isinstance(task_ids, np.ndarray):
                self._results = self._results[self._results['Id'].isin([task_ids.tolist()])]
            else:
                raise UserWarning(f"Task_ids should be of type <int> or <list> or <numpy.ndarray>, found: {type(task_ids)}")

        # print('### Number of samples:', len(self._results))

        # desired_gc = 'gc' in self.task.value
        constrained = 'constrained' in self.task.value

        if folding_algorithm is None:
            folding_algorithm = RNAFold()

        try:
            folding_algo_name = folding_algorithm.__name__()
        except:
            folding_algo_name = 'custom'

        self._results[['predicted_sequence', 'time']] = self._results.apply(lambda x: self._predict(x, wrapper_function, *args, **kwargs), axis=1, result_type='expand')
        if folding_algorithm.__name__() == 'RNAFold':
            self._results[['predicted_pairs', 'energy']] = self._results.apply(lambda x: folding_algorithm(x.predicted_sequence), axis=1, result_type='expand')
        else:
            self._results.loc[:, 'predicted_pairs'] = self._results.apply(lambda x: folding_algorithm(x.predicted_sequence), axis=1)
        self._results = self._results.dropna()

        #predicted df, get predicted pairs and sequences and remove predicted substring
        predicted_columns = [col for col in self._results.columns if col.startswith('predicted_') or col == 'Id']
        pred_df = self._results.filter(predicted_columns)
        # Create a dictionary to map old column names to new column names
        column_mapping = {col: col.replace('predicted_', '') for col in predicted_columns}
        # Create a new dataframe with renamed columns
        pred_df = pred_df.rename(columns=column_mapping)
        #get column names before
        old_cols = pred_df.columns
        if self.feature_extractors is not None:
            pred_df = self.extract_features(pred_df)
        #drop old columns
        pred_df = pred_df.drop(columns=old_cols)
        #rename columns with predicted_ prefix
        pred_df.columns = ['predicted_' + col for col in pred_df.columns]
        #add pred_df columns to self._results if not already present
        self._results = pd.concat([self._results, pred_df], axis=1)
        self._results.loc[:, 'per_sample_metrics'] = self._results.apply(lambda x: self.gd_metrics(x,
                                                                                                desired_gc=desired_gc,
                                                                                                gc_tolerance=gc_tolerance,
                                                                                                constrained=constrained,
                                                                                            ), axis=1)

        per_sample_metrics = pd.DataFrame.from_dict(self._results['per_sample_metrics'].to_list())
        per_sample_metrics.index = self._results.index

        # requires workaround: unsure where wl comes from from tiem to time... TODO: find out and debug again!
        if 'wl' in per_sample_metrics.columns:
            per_sample_metrics.drop('wl', axis=1, inplace=True)

        self._results = pd.concat([self._results, per_sample_metrics], axis=1)
        self.df_results = self.dl_metrics.evaluate(self._results)
        per_sample_metrics = per_sample_metrics.fillna(0)

        scores = {m: np.mean(per_sample_metrics[m])
                         if not 'solved' in m or not 'valid' in m else np.sum(per_sample_metrics[m])
                         for m in per_sample_metrics.columns}
        # append df_results to scores
        scores.update({m: self.df_results[m] for m in self.df_results})

        if save_results:
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            Path(self.results_dir, 'RNA_design', self.task.value, algorithm_name, folding_algo_name).mkdir(exist_ok=True, parents=True)

            if results_path is not None:
                out_path = Path(results_path)
            else:
                out_path = Path(self.results_dir, 'RNA_design', self.task.value, algorithm_name, folding_algo_name, f"{'_'.join([str(x) for x in [algorithm_name, self._evaluation_counter, 'min_len', self.min_length, 'max_len', self.max_length, 'nc', self.nc, 'pks', self.pks, 'multiplets', self.multiplets, folding_algorithm, 'timeout', self.timeout]])}_{'-'.join(now.split('/')).replace(' ', '-')}.plk")
            with open(out_path, 'wb') as f:
                pickle.dump(self._results, f)

        return {m: np.round(v, 3) for m, v in scores.items()}

    @timing
    def _predict(self, x, wrapper_function, *args, **kwargs):
        return wrapper_function(x, *args, **kwargs)
        # return limited_execution(timing(wrapper_function),
        #                          x,
        #                          *args,
        #                          # cpu_time=self.timeout,
        #                          wall_time=self.timeout,
        #                          **kwargs,
        #                          )


class RiboswitchDesignBenchmark(Benchmark):
    _metrics = ['has_aptamer', 'has_8-U', 'valid_shape', 'valid_aptamer_hairpin',
                '8-U_unpaired', 'valid_co_fold']

    __div_metrics: List[str] = ["diversity", "diameter", "DPP", "sum_btlnk"]
    __gen_dists: List[str] = ["hamming", ]  # "lv"]
    __feats: List[str] = ["sequence", "structure"]
    __feat_distmap: Dict[str, List[str]] = {"predicted_sequence": ["hamming"],  # , "lv"],  # "lv"],
                                            "predicted_s_0": ["l2"],
                                            "predicted_h_0": ["l2"],
                                            "predicted_i_0": ["l2"],
                                            "predicted_i_1": ["l2"],
                                            # "predicted_pairs": ["wl_pairs"],
                                            }
    _evaluation_counter = 0

    def __init__(self,
                 data_dir: str = 'data',
                 results_dir: str = 'results',
                 feature_extractors: Optional[Dict[str, Callable]] = {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")},
                 ):
        self.task = Tasks('riboswitch_design')
        self.metrics = RiboswitchDesignMetrics(self._metrics)
        super().__init__(
            data_dir=data_dir,
            results_dir=results_dir,
            nc=True,
            pks=True,
            multiplets=True,
            min_length=None,
            max_length=None,
            feature_extractors=feature_extractors
        )

        self.dl_metrics = DistributionLearningMetrics(general_dists=self.__gen_dists, nov_metrics=None, div_metrics=self.__div_metrics, feats=self.__feats, feat_spec_dists=self.__feat_distmap)
        self.feature_extractors = feature_extractors
    def __call__(self,
                 wrapper_function,
                 *args,
                 save_results=True,
                 desired_gc=None,
                 gc_tolerance=0.01,
                 desired_energy=None,
                 energy_tolerance=0.1,
                 results_path=None,
                 algorithm_name='custom',
                 **kwargs,
                 ):
        self._evaluation_counter += 1
        folding_algorithm = RNAFold()
        predictions = wrapper_function(*args, **kwargs)

        self._results = pd.DataFrame.from_dict([{'Id': i, 'predicted_sequence': s}
                                                for i, s in enumerate(predictions)])
        self._results[['predicted_pairs', 'energy']] = self._results.apply(
            lambda x: folding_algorithm(x.predicted_sequence), axis=1, result_type='expand')

        self._results['per_sample_metrics'] = self._results.apply(lambda x: self.metrics(x,
                                                                                         desired_gc=desired_gc,
                                                                                         desired_energy=desired_energy,
                                                                                         gc_tolerance=gc_tolerance,
                                                                                         energy_tolerance=energy_tolerance,
                                                                                         ), axis=1)

        per_sample_metrics = pd.DataFrame.from_dict(self._results['per_sample_metrics'].to_list())
        per_sample_metrics.index = self._results.index

        self._results = pd.concat([self._results, per_sample_metrics], axis=1)

        scores = {m: np.sum(per_sample_metrics[m]) / len(per_sample_metrics)
                  for m in per_sample_metrics.columns if 'valid' in m}

        valid_candidates = self._results[
            self._results['valid_sequence_and_structure'] & self._results['valid_gc_content'] & self._results[
                'valid_energy']]

        if len(valid_candidates) > 0:
            scores['unique_valid_candidates'] = len(valid_candidates['sequence'].unique()) / len(
                valid_candidates)
            scores['unique_valid_structures'] = len(valid_candidates['structure'].unique()) / len(
                valid_candidates)
        else:
            scores['unique_valid_candidates'] = 0.0
            scores['unique_valid_structures'] = 0.0

        scores['valid_candidates'] = len(valid_candidates) / len(self._results)


        predicted_columns = [col for col in self._results.columns if col.startswith('predicted_') or col == 'Id']
        pred_df = self._results.filter(predicted_columns)
        # Create a dictionary to map old column names to new column names
        column_mapping = {col: col.replace('predicted_', '') for col in predicted_columns}
        # Create a new dataframe with renamed columns
        pred_df = pred_df.rename(columns=column_mapping)
        # get column names before
        old_cols = pred_df.columns
        pred_df = self.extract_features(pred_df)
        # drop old columns
        pred_df = pred_df.drop(columns=old_cols)
        # rename columns with predicted_ prefix
        pred_df.columns = ['predicted_' + col for col in pred_df.columns]
        # add pred_df columns to self._results if not already present
        self._results = pd.concat([self._results, pred_df], axis=1)

        self.df_results = self.dl_metrics.evaluate(self._results)

        scores.update({m: self.df_results[m] for m in self.df_results})

        if save_results:
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            Path(self.results_dir, 'Riboswitch_design', algorithm_name).mkdir(exist_ok=True, parents=True)

            if results_path is not None:
                out_path = Path(results_path)
            else:
                out_path = Path(self.results_dir, 'Riboswitch_design', algorithm_name,
                                f"{'_'.join([str(x) for x in [algorithm_name, self._evaluation_counter, 'gc', desired_gc, 'energy', desired_energy]])}_{'-'.join(now.split('/')).replace(' ', '-')}.plk")
            with open(out_path, 'wb') as f:
                pickle.dump(self._results, f)

        return {m: np.round(v, 3) for m, v in scores.items()}


class CompartiveBenchmark(Benchmark):

    __nov_metrics: List[str] = ["iou", "novelty"]
    __div_metrics: List[str] = ["diversity", "diameter"]
    __gen_dists: List[str] = ["hamming", ]  # "lv"]
    __feats: List[str] = ["sequence", "structure"]
    __feat_distmap: Dict[str, List[str]] = {"sequence": ["hamming", "hamming_nan"],  # "lv"],
                                            "s_0": ["l2"],
                                            "h_0": ["l2"],
                                            "i_0": ["l2"],
                                            "i_1": ["l2"],
                                            #"structure": ["hamming"],
                                            "pairs": ["wl_pairs"],
                                            }
    __kl_feats: List[str] = [ "gc_content", "length", "energy"] #  "s_0", "h_0", "i_0", "i_1", "energy"]
    _evaluation_counter = 0

    def __init__(self,
                 data_dir: str = 'data',
                 results_dir: str = 'results',
                 ):
        self.task = Tasks('riboswitch_design')

        super().__init__(
            data_dir=data_dir,
            results_dir=results_dir,
            nc=True,
            pks=True,
            multiplets=True,
            min_length=None,
            max_length=None,
        )

        self.dl_metrics = DistributionLearningComparativeMetrics(general_dists=self.__gen_dists, nov_metrics=self.__nov_metrics, div_metrics=self.__div_metrics, kl_feats=self.__kl_feats, feat_spec_dists=self.__feat_distmap)

    def __call__(self,
                 orig_data: pd.DataFrame,
                 gen_data: pd.DataFrame,
                 *args,
                 save_results=True,
                 desired_gc=None,
                 gc_tolerance=0.01,
                 desired_energy=None,
                 energy_tolerance=0.1,
                 results_path=None,
                 algorithm_name='custom',
                 **kwargs,
                 ):
        self._evaluation_counter += 1

        column_mapping = {col: col.replace('predicted_', '') for col in gen_data.columns}
        # Create a new dataframe with renamed columns
        self.orig_data = orig_data
        self.gen_data = gen_data.rename(columns=column_mapping)

        self.df_results = self.dl_metrics.evaluate(self.orig_data, self.gen_data)

        if save_results:
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            Path(self.results_dir, 'Riboswitch_design_compare', algorithm_name).mkdir(exist_ok=True, parents=True)

            if results_path is not None:
                out_path = Path(results_path)
            else:
                out_path = Path(self.results_dir, 'Riboswitch_design_compare', algorithm_name,
                                f"{'_'.join([str(x) for x in [algorithm_name, self._evaluation_counter, 'gc', desired_gc, 'energy', desired_energy]])}_{'-'.join(now.split('/')).replace(' ', '-')}.plk")
            with open(out_path, 'wb') as f:
                pickle.dump(self.df_results, f)

        return {m: np.round(v, 3) for m, v in self.df_results.items()}


class RnaFoldingBenchmark(Benchmark):

    _gd_metrics: List[str] = [
                 'f1_score',
                 'mcc',
                 'wl',
                 'recall',
                 'precision',
                 'specificity',
                 'solved',
                 'f1_shifted',
                 ]
    __nov_metrics: List[str] = ["iou", "novelty"]
    __div_metrics: List[str] = ["diversity", "diameter", "DPP", "sum_btlnk"]
    __gen_dists: List[str] = ["hamming", ] # "lv"]
    __feats: List[str] = ["sequence", "structure"]
    __feat_distmap: Dict[str, List[str]] = {"predicted_sequence": ["hamming", ],  #"lv"],
                                            "predicted_s_0" : ["l2"],
                                            "predicted_h_0" : ["l2"],
                                            "predicted_i_0" : ["l2"],
                                            "predicted_i_1": ["l2"],
                                            #"structure": ["hamming"],
                                            "predicted_pairs": ["wl_pairs"],
                                            }


    _evaluation_counter = 0

    def __init__(
                 self,
                 task : str = 'inter_family',
                 nc: bool = True,
                 pks: bool = True,
                 multiplets: bool = True,
                 data_dir: str = 'data',
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 results_dir: str = 'results',
                 feature_extractors: Optional[Dict[str, Callable]] = {'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="motif_lists")},
                 ):

        # get task
        try:
            self.task = Tasks(task.lower())
        except ValueError as e:
            valid_tasks = [x.value for x in Tasks]
            raise ValueError(f"Unknown value of parameter 'task': '{task}'."
                                 f"Please use one of {valid_tasks}.") from e

        self.gd_metrics = GoalDirectedMetrics(self._gd_metrics)
        self.dl_metrics = DistributionLearningMetrics(general_dists=self.__gen_dists, nov_metrics=self.__nov_metrics, div_metrics=self.__div_metrics, feats=self.__feats, feat_spec_dists=self.__feat_distmap)
        self.feature_extractors = feature_extractors

        super().__init__(
                         min_length=min_length,
                         max_length=max_length,
                         nc=nc,
                         pks=pks,
                         multiplets=multiplets,
                         results_dir=results_dir,
                         data_dir=data_dir,
                        )
        self.load_data()


    def __call__(self,
                 wrapper_function,
                 *args,
                 save_results=True,
                 dataset=None,
                 results_path=None,
                 task_ids=None,
                 algorithm_name='custom',
                 dist_metrics = False,
                 **kwargs,
                 ):
        self._evaluation_counter += 1
        if dataset is not None:
            if isinstance(dataset, RnaDataset):
                self._results = dataset.data.copy()
            elif isinstance(dataset, pd.DataFrame):
                self._results = dataset.copy()
            elif isinstance(dataset, TorchDataset):
                self._results = dataset.dataset.data.copy()
            else:
                raise UserWarning(f"Dataset type {type(dataset)} currently not supported")
            if self.feature_extractors is not None:
                # check if features are already extracted
                self._results = self.extract_features(self._results)


        else:
            self._results = self.data.copy()
        # print(self._results.columns)
        if task_ids is not None:
            if isinstance(task_ids, int):
                self._results = self._results[self._results['Id'].isin([task_ids])]
            elif isinstance(task_ids, list):
                self._results = self._results[self._results['Id'].isin(task_ids)]
            elif isinstance(task_ids, np.ndarray):
                self._results = self._results[self._results['Id'].isin([task_ids.tolist()])]
            else:
                raise UserWarning(f"Task_ids should be of type <int> or <list> or <numpy.ndarray>, found: {type(task_ids)}")

        self._results.loc[:, 'predicted_sequence'] = self._results['sequence']

        self._results[['predicted_pairs', 'time']] = self._results.apply(lambda x: self._predict(x, wrapper_function, *args, **kwargs), axis=1, result_type='expand')
        # TODO: Fix distribution learning when no pairs are provided
        # Current workaround:
        # drop all samples without pairs
        self._results = self._results[self._results['pairs'].apply(lambda x: x != [])]
        predicted_columns = [col for col in self._results.columns if col.startswith('predicted_') or col == 'Id']
        pred_df = self._results.filter(predicted_columns)

        # Create a dictionary to map old column names to new column names
        column_mapping = {col: col.replace('predicted_', '') for col in predicted_columns}
        # Create a new dataframe with renamed columns
        pred_df = pred_df.rename(columns=column_mapping)
        #get column names before
        old_cols = pred_df.columns
        if self.feature_extractors is not None:
            pred_df = self.extract_features(pred_df)
        #drop old columns
        pred_df = pred_df.drop(columns=old_cols)
        #rename columns with predicted_ prefix
        pred_df.columns = ['predicted_' + col for col in pred_df.columns]
        #add pred_df columns to self._results if not already present
        self._results = pd.concat([self._results, pred_df], axis=1)
        self._results.loc[:, 'per_sample_metrics'] = self._results.apply(lambda x: self.gd_metrics(x), axis=1)
        #self.df_results =

        per_sample_metrics = pd.DataFrame.from_dict(self._results['per_sample_metrics'].to_list())
        per_sample_metrics.index = self._results.index

        # requires workaround: unsure where wl comes from from tiem to time... TODO: find out and debug again!
        if 'wl' in per_sample_metrics.columns:
            per_sample_metrics.drop('wl', axis=1, inplace=True)

        self._results = pd.concat([self._results, per_sample_metrics], axis=1)
        if dist_metrics:
            self.df_results = self.dl_metrics.evaluate(self._results)
        
        per_sample_metrics = per_sample_metrics.fillna(0)

        scores = {m: np.mean(per_sample_metrics[m])
                         if not 'solved' in m or not 'valid' in m else np.sum(per_sample_metrics[m])
                         for m in per_sample_metrics.columns}
        if dist_metrics:
            # append df_results to scores
            scores.update({m: self.df_results[m] for m in self.df_results})

        if save_results:
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            Path(self.results_dir, "RNA_folding", self.task.value, algorithm_name).mkdir(exist_ok=True, parents=True)

            if results_path is not None:
                out_path = Path(results_path)
            else:
                out_path = Path(self.results_dir, "RNA_folding", self.task.value, algorithm_name, f"{'_'.join([str(x) for x in [algorithm_name, self._evaluation_counter, 'min_len', self.min_length, 'max_len', self.max_length, 'nc', self.nc, 'pks', self.pks, 'multiplets', self.multiplets]])}_{'-'.join(now.split('/')).replace(' ', '-')}.plk")
            with open(out_path, 'wb') as f:
                pickle.dump(self._results, f)

        return {m: np.round(v, 3) for m, v in scores.items()}

    @timing
    def _predict(self, x, wrapper_function, *args, **kwargs):
        return wrapper_function(x, *args, **kwargs)
