import typing
import dask.dataframe as dd

import numpy as np
import pandas as pd
import torch
import numpy as np
import pickle
# import modin
import pandas as pd
# import pandas as pd

from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Callable, Union, Dict

from RnaBench.download import select_and_download
from RnaBench.lib.utils import posencoding2int, pos2pairs, pairs2db
from RnaBench.lib.tasks import RNA, TaskFactory
from RnaBench.lib.feature_extractors import *
from RnaBench.lib.alphabets import (
    AlphabetConverter,
    Nucleotide,
    Structure,
    SpecialSymbols,
    get_nuc_vocab,
    get_struc_vocab,
)

from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore")


class TorchDataset(Dataset):
    def __init__(self,
                 dataset,
                 task,
                 padding='end',
                 seed=0,
                 matrix=False,
                 transform=None,
                 ):

        self.dataset = dataset
        self.transform = transform
        self.matrix = matrix

        self.tasks = TaskFactory(
            dataset=dataset,
            seed=seed,
            task=task,
            padding=padding,
            numpy=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.tasks[idx]

        if self.transform:
            for trfm in self.transform:
                sample = trfm(sample)
        return sample.to_dict()

    @property
    def seq_itos(self):
        return self.tasks.seq_itos

    @property
    def seq_stoi(self):
        return self.tasks.seq_stoi

    @property
    def struc_itos(self):
        return self.tasks.struc_itos

    @property
    def struc_itos(self):
        return self.tasks.struc_stoi


class ToTensor():
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return sample.to_torch(device=self.device)


class RnaBenchmarkDataset():
    def __init__(self,
                 dataset: str,
                 nc: bool = True,
                 pks: bool = True,
                 multiplets: bool = True,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 matrix: Optional[bool] = None,
                 ):
        self._nc = nc
        self._pks = pks
        self._multiplets = multiplets
        self._min_length = min_length
        self._max_length = max_length
        self._matrix = matrix

        self.data = RnaDataset.load(dataset=dataset,
                                    nc=self._nc,
                                    pks=self._pks,
                                    multiplets=self._multiplets,
                                    min_length=self._min_length,
                                    max_length=self._max_length,
                                    )

    def __iter__(self):
        yield from map(RnaDataset.to_rna, (list(tup)+[self._matrix] for tup in zip(self.data['Id'],
                                               self.data['sequence'],
                                               self.data['pairs'],
                                               self.data['gc_content'],
                                               self.data['length'],
                                               )))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        d = self.data.iloc[idx, :]

        return RnaDataset.to_rna((idx, d['sequence'], d['pairs'], d['gc_content'], d['length'], self._matrix))

    def __len__(self):
        return len(self.data)


class RnaDataset():
    def __init__(self,
                 dataset: Union[Path, pd.DataFrame],
                 nc: bool = True,
                 pks: bool = True,
                 multiplets: bool = True,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 sequence_vocab: Optional[Tuple[str]] = Nucleotide.iupac_alphabet,
                 structure_vocab: Optional[Tuple[str]] = Structure.extended_dot_bracket,
                 feature_extractors: Optional[Dict[str, Callable]] = None,
                 matrix: Optional[bool] = None,
                 ):
        
        self._nc = nc
        self._pks = pks
        self._multiplets = multiplets
        self._min_length = min_length
        self._max_length = max_length
        self._matrix = matrix

        if not isinstance(dataset, pd.DataFrame):
            self.data = RnaDataset.load(dataset=dataset,
                                        nc=self._nc,
                                        pks=self._pks,
                                        multiplets=self._multiplets,
                                        min_length=self._min_length,
                                        max_length=self._max_length,
                                        )
        else:
            self.data = dataset

        self._max_seq_length = self.data['length'].max()
        self.data.loc[:, 'pair_length'] = self.data['pairs'].apply(len)
        self.max_pair_length = self.data['pair_length'].max()

        if sequence_vocab is None:
            self.sequence_vocab = get_nuc_vocab(self.data)
        else:
            self.sequence_vocab = sequence_vocab
        if structure_vocab is None:
            self.structure_vocab = get_struc_vocab(self.data)
        else:
            self.structure_vocab = structure_vocab

        self.seq_alphabet_converter = AlphabetConverter(self.sequence_vocab)
        self.struc_alphabet_converter = AlphabetConverter(self.structure_vocab)

        if feature_extractors is not None:
            self.feature_extractors = feature_extractors
            self.distribution = None
            self.descriptors_distribution = {}
            self.data = self.extract_features(self.data, self.feature_extractors)
            # print(self.data)

    @staticmethod
    def load(
            dataset: Path,
            nc: bool,
            pks: bool,
            multiplets: bool,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            feature_extractors: Optional[Dict[str, Callable]] = None,
    ):
        if not Path(dataset).is_file():
            # Path(self.data_dir).mkdir(exist_ok=True, parents=True)
            t = '_'.join(str(Path(dataset).stem).split('_')[:-1])
            select_and_download(task=t,
                                save_dir='data',
                                )

        data = pd.read_pickle(dataset)

        if min_length is not None:
            data = data[data['sequence'].apply(lambda x: min_length <= len(x))]
        if max_length is not None:
            data = data[data['sequence'].apply(lambda x: len(x) <= max_length)]

        if not nc:
            data = data[data['has_nc'] == False] if 'has_nc' in data.columns else data

        if not pks:
            data = data[data['has_pk'] == False] if 'has_pk' in data.columns else data

        if not multiplets:
            data = data[data['has_multiplet'] == False] if 'has_multiplet' in data.columns else data

        if feature_extractors is not None:
            data = RnaDataset.extract_features(data, feature_extractors)
        return data

    @staticmethod
    def to_rna(l):
        return RNA(rna_id=l[0],
                   sequence=l[1],
                   # seq_length=len(l[1]),
                   pairs=l[2],
                   gc=l[3],
                   length=l[4],
                   # structure=l[2],
                   # pos1id=l[3],
                   # pos2id=l[4],
                   # pk=l[5],
                   matrix=l[5],
                   )

    def __iter__(self):
        yield from map(RnaDataset.to_rna, (list(tup) + [self._matrix] for tup in zip(self.data['Id'],
                                                                                     self.data['sequence'],
                                                                                     self.data['pairs'],
                                                                                     self.data['gc_content'],
                                                                                     self.data['length'],
                                                                                     # self.data['pk'],
                                                                                     )))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        d = self.data.iloc[idx, :]

        return RnaDataset.to_rna((d['Id'], d['sequence'], d['pairs'], d['gc_content'], d['length'], self._matrix))

    def __len__(self):
        return len(self.data)

    def select_by_origin(self, origins: list):
        data = pd.concat([self.data[self.data['origin'].str.contains(part)]
                          for part in partitions])
        return RnaDataset(data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix,
                          )

    @property
    def seq_stoi(self):
        return self.seq_alphabet_converter.stoi

    @property
    def struc_stoi(self):
        return self.struc_alphabet_converter.stoi

    @property
    def seq_itos(self):
        return self.seq_alphabet_converter.itos

    @property
    def struc_itos(self):
        return self.struc_alphabet_converter.itos

    @property
    def train(self):
        return RnaDataset(self.train_data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix,
                          )

    @property
    def valid(self):
        return RnaDataset(self.validation_data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix
                          )

    @property
    def test(self):
        return RnaDataset(self.test_data,
                          sequence_vocab=self.sequence_vocab,
                          structure_vocab=self.structure_vocab,
                          matrix=self._matrix
                          )

    @property
    def train_data(self):
        return self.data[self.data['origin'].str.contains('train')]

    @property
    def validation_data(self):
        return self.data[self.data['origin'].str.contains('valid')]

    @property
    def test_data(self):
        return self.data[~(self.data['origin'].str.contains('valid') | self.data['origin'].str.contains('train'))]

    @property
    def has_train(self):
        return not self.train_data.empty

    @property
    def has_validation(self):
        return not self.validation_data.empty

    @property
    def has_test(self):
        return not self.test_data.empty

    @property
    def max_seq_length(self):
        return self._max_seq_length

    @staticmethod
    def extract_features(data, feature_extractors=None, update=False):
        means, scales = [], []

        fe = feature_extractors

        if fe is not None:
            for feat, feat_ext in fe.items():
                # print(f"extracting feature {feat}")
                # TODO(KARIM): add a mapping between type of distribution string to actual distribution class
                # if feat in self.descriptors_distribution.keys():
                #     # print(f" feature {feat} exists in descriptor distribution")
                #     pass

                # agg_level = feat_ext.aggregation_level if hasattr(feat_ext, "aggregation_level") else "strand"
                # feats = []

                feat_cols = feat_ext.feat_names if hasattr(feat_ext, "feat_names") else [feat]
                if feat not in data:
                    feat_extracted = data.apply(feat_ext, axis=1, result_type="expand")
                    feat_extracted.columns = feat_cols
                    data.loc[:, feat_cols] = feat_extracted.to_numpy()
                # if agg_level == 'strand':
                #     if feat not in data:
                #         feat_extracted = data.apply(feat_ext, axis=1, result_type="expand")
                #         feat_extracted.columns = feat_cols
                #         #print(feat_extracted)
                #         #print(feat_cols)
                #         data.loc[:, feat_cols] = feat_extracted.to_numpy()#.reset_index(drop=True)
                #     mean = data[feat_cols].mean()
                #     std = data[feat_cols].std()+1e-5
                #     for sub_feat in feat_cols:
                #         col = f"{feat}_{sub_feat}" if feat != sub_feat else feat
                #         if not np.isnan(mean[sub_feat]) and not np.isnan(std[sub_feat]):
                #             means.append(mean[sub_feat])
                #             scales.append(std[sub_feat])
                #             #self.descriptors_distribution[col] = Normal(mean[sub_feat], std[sub_feat])
                #
                # else:
                #     feat_extracted = data.apply(feat_ext, axis=1)
                #     data.loc[:, feat_cols] = feat_extracted.to_numpy()
                #     feat_extracted = pd.concat(feat_extracted.to_list())
                #     grouped_feats = feat_extracted.drop(labels=['rna_id', 'motif_idx'], axis = 1).groupby("type")
                #     mean = grouped_feats.mean()
                #     std = grouped_feats.std()+1e-5
                #     for sub_feat in feat_cols:
                #         t, n = sub_feat.split("_")
                #         n= int(n)
                #         if not np.isnan(mean.loc[t, n]) and not np.isnan(std.loc[t, n]):
                #             means.append(mean.loc[t, n])
                #             scales.append(std.loc[t, n])
                #             #self.descriptors_distribution[f"{feat}_{sub_feat}"] = Normal(mean.loc[t, n], std.loc[t, n])

            return data
            # self.distribution = MultivariateNormal(torch.tensor(means), torch.tensor(np.diag(scales)))

    def extract_features_ds(self, feature_extractors=None, update=False):
        means, scales = [], []

        if hasattr(self, "feature_extractors") and feature_extractors is None:
            fe = self.feature_extractors
        else:
            fe = feature_extractors

        if fe is not None:
            for feat, feat_ext in fe.items():
                # print(f"extracting feature {feat}")
                # TODO(KARIM): add a mapping between type of distribution string to actual distribution class
                # if feat in self.descriptors_distribution.keys():
                #     # print(f" feature {feat} exists in descriptor distribution")
                #     pass

                agg_level = feat_ext.aggregation_level if hasattr(feat_ext, "aggregation_level") else "strand"
                feat_cols = feat_ext.feat_names if hasattr(feat_ext, "feat_names") else [feat]

                if agg_level == 'strand':
                    if feat not in data:
                        feat_extracted = data.apply(feat_ext, axis=1, result_type="expand")
                        feat_extracted.columns = feat_cols
                        # print(feat_extracted)
                        # print(feat_cols)
                        data.loc[:, feat_cols] = feat_extracted.to_numpy()  # .reset_index(drop=True)
                    mean = data[feat_cols].mean()
                    std = data[feat_cols].std() + 1e-5
                    for sub_feat in feat_cols:
                        col = f"{feat}_{sub_feat}" if feat != sub_feat else feat
                        if not np.isnan(mean[sub_feat]) and not np.isnan(std[sub_feat]):
                            means.append(mean[sub_feat])
                            scales.append(std[sub_feat])
                            self.descriptors_distribution[col] = Normal(mean[sub_feat], std[sub_feat])

                else:
                    feat_extracted = data.apply(feat_ext, axis=1)
                    data.loc[:, feat_cols] = feat_extracted.to_numpy()
                    feat_extracted = pd.concat(feat_extracted.to_list())
                    grouped_feats = feat_extracted.drop(labels=['rna_id', 'motif_idx'], axis=1).groupby("type")
                    mean = grouped_feats.mean()
                    std = grouped_feats.std() + 1e-5
                    for sub_feat in feat_cols:
                        t, n = sub_feat.split("_")
                        n = int(n)
                        if not np.isnan(mean.loc[t, n]) and not np.isnan(std.loc[t, n]):
                            means.append(mean.loc[t, n])
                            scales.append(std.loc[t, n])
                            self.descriptors_distribution[f"{feat}_{sub_feat}"] = Normal(mean.loc[t, n], std.loc[t, n])

            self.distribution = MultivariateNormal(torch.tensor(means), torch.tensor(np.diag(scales)))
            return data

    def dask(self, n_workers=4):
        self.data = dd.from_pandas(self.data, npartitions=n_workers)
        return self

    def to_pandas(self):
        if isinstance(self.data, dd.DataFrame):
            self.data = self.data.compute()
        return self


if __name__ == '__main__':
    # get dataset as a argument
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='inverse_rna_folding_benchmark')
    parser.add_argument('--data-dir', type=str, default='data')

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    data_path = Path(f'{data_dir}/{dataset}')

    # dataset = 'inverse_rna_folding_benchmark'
    data = RnaDataset.load(data_path,
                           nc=False,
                           pks=False,
                           multiplets=False,
                           feature_extractors={
                               'structural_motifs': StructuralMotifs(source="forgi", aggregation_level="dataset")})

    # save dataframe in a pickle file in directory data_preprocessed

    # make sure to create the directory data_preprocessed before running the code
    # make directory data_preprocessed in the root directory of the project
    Path('data_preprocessed', *data_path.parts[:-1]).mkdir(parents=True, exist_ok=True)
    # change only the first dir in the path to data_preprocessed
    data_save_path = Path("data_preprocessed", *data_path.parts)
    data.to_pickle(f'{str(data_save_path)}')

    # print(data.distribution.loc, data.distribution.covariance_matrix)
    # print(data.data.head())