import itertools
import torch
import RNA
import subprocess
import numpy as np
import copy
import pandas as pd
from typing import List
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, ShortestPath
from scipy.spatial.distance import cdist
from torch.distributions import kl_divergence, Normal, MultivariateNormal, Bernoulli
from nltk import edit_distance
from scipy import signal
from pathlib import Path

from RnaBench.lib.utils import db2mat, pairs2db, pairs2mat, db2pairs


def get_metric(m):
    """
    Returns performance measure functions.
    """
    if m == 'f1_score':
        return f1
    elif m == 'mcc':
        return mcc
    elif m == 'wl':
        return weisfeiler_lehman_score_from_string
    elif m == 'hamming_structure':
        return hamming_distance_score
    elif m == 'recall':
        return recall
    elif m == 'precision':
        return precision
    elif m == 'specificity':
        return specificity
    elif m == 'pair_ratio':
        return abs_pair_ratio_score
    elif m == 'solved':
        return solved_from_string
    elif m == 'f1_shifted':
        return evaluate_shifted_f1
    elif m == 'gc_content_absolute':
        return gc_content_abs_score
    elif m == 'hamming_sequence':
        return rel_hamming_list_score
    elif m == 'valid_shape':
        return eval_shape
    elif m == 'valid_aptamer_hairpin':
        return check_aptamer_structure
    elif m == 'has_aptamer':
        return has_aptamer
    elif m == 'has_8-U':
        return has_8_U
    elif m == '8-U_unpaired':
        return check_8_u
    elif 'valid_co_fold':
        return evaluate_co_transcriptional_folding_simulation


def get_general_dist(m):
    if m == 'hamming':
        return lambda s1, s2: hamming_distance_relative(s1, s2, None)
    if m == 'hamming_nan':
        return lambda s1, s2: hamming_distance_relative(s1, s2, None, len_mismatch_nan=True)
    elif m == "lv":
        return edit_distance
    elif m == 'wl':
        return weisfeiler_lehman_score_from_string
    elif m == 'wl_pairs':
        return graph_distance_score_from_pairs
    elif m == "l1":
        return lambda s1, s2: np.linalg.norm(s1 - s2, ord=1)
    elif m == "l2":
        return "euclidean"  # lambda s1, s2: np.linalg.norm(s1 - s2, ord=2)


class BaseMetrics():
    """
    Base class for all metrics.
    """
    def __init__(self, metrics):
        for att, value in metrics.items():
            setattr(self, att, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        for att in vars(self):
            yield att, self[att]

    def items(self):
        for att in self:
            yield att, self[att]

    def values(self):
        for att in self:
            yield self[att]

    def keys(self):
        for att in self:
            yield att

    def __call__(self, row, *args, **kwargs):
        return self.evaluate(row, *args, **kwargs)

class GoalDirectedMetrics(BaseMetrics):
    """
    Evaluated metrics that can be used for goal-oriented design.
    """

    def __init__(self, metrics):
        """
        Initialize desired metrics.
        """
        metrics = {k: get_metric(k) for k in metrics}
        super().__init__(metrics)

    def evaluate(self,
                 row,
                 desired_gc=None,
                 gc_tolerance=0.01,
                 constrained=False,
                 ):
        """
        Evaluate all metrics.
        Called on a datafram row.

        """
        if isinstance(row['predicted_sequence'], float) or isinstance(row['predicted_pairs'], float):
            return {m: np.nan for m, _ in self}
        else:
            results = {}

            sequence = row['sequence']
            pred_sequence = row['predicted_sequence']

            if desired_gc is not None:
                # gc = row['gc_content']
                gc = desired_gc
                min_gc = gc - gc_tolerance
                max_gc = gc + gc_tolerance
                pred_gc = (''.join(pred_sequence).count('G') + ''.join(pred_sequence).count('C')) / len(pred_sequence)
                results['valid_gc_content'] = min_gc < pred_gc < max_gc
                results['gc_score'] = 1 - abs(gc - pred_gc)

            if constrained:
                true_seq = np.asarray(sequence)
                pred_seq = np.asarray(pred_sequence)
                d = (true_seq != pred_seq).astype(np.int8)
                ignore = (true_seq != 'N').astype(np.int8)
                distance = np.sum(d * ignore)
                results['valid_sequence_constraints'] = distance == 0
                results['sequence_distance_score'] = 1 - (distance / len(sequence))

            true_mat = pairs2mat(row['pairs'], length=row['length'])
            pred_mat = pairs2mat(row['predicted_pairs'], length=row['length'])

            tp = tp_from_matrices(pred_mat, true_mat)
            fp = get_fp(pred_mat, tp)
            fn = get_fn(true_mat, tp)
            tn = tn_from_matrices(pred_mat, true_mat)

            for name, metric in self:
                if name == 'gc':
                    continue
                if name == 'wl':
                    results['weisfeiler_lehman'] = graph_distance_score_from_matrices(
                        pred_mat,
                        true_mat,
                        kernel='WeisfeilerLehman'
                    )
                elif name == 'solved':
                    results['solved'] = solved_from_mat(pred_mat, true_mat)
                elif name == 'f1_shifted':
                    results['f1_shifted'] = evaluate_shifted_f1(pred_mat, true_mat)
                elif name == 'gc_content':
                    continue
                    # results['gc_content_score'] = gc_content_abs_score(row['sequence'], row['predicted_sequence'])
                elif name == 'hamming_sequence':
                    continue
                    # results['hamming_sequence'] = rel_hamming_list_score(row['sequence'], row['predicted_sequence'])
                else:
                    results[name] = metric(tp, fp, tn, fn)
        results['time'] = row['time']
        return results


class RiboswitchDesignMetrics(BaseMetrics):
    def __init__(self, metrics):
        metrics = {k: get_metric(k) for k in metrics}

        super().__init__(metrics)

    def evaluate(
            self,
            row,
            desired_gc=None,
            gc_tolerance=0.01,
            desired_energy=None,
            energy_tolerance=0.1,
    ):

        sequence = ''.join(row['predicted_sequence'])
        structure = pairs2db(row['predicted_pairs'], sequence)
        energy = row['energy']
        gc = (sequence.count('G') + sequence.count('C')) / len(sequence)

        results = {}

        results['gc_content'] = gc
        results['structure'] = structure
        results['sequence'] = sequence

        if desired_gc is not None:
            min_gc = desired_gc - gc_tolerance
            max_gc = desired_gc + gc_tolerance
            results['valid_gc_content'] = min_gc < gc < max_gc
        else:
            results['valid_gc_content'] = True
        if desired_energy is not None:
            min_energy = desired_energy - energy_tolerance
            max_energy = desired_energy + energy_tolerance
            results['valid_energy'] = min_energy < energy < max_energy
        else:
            results['valid_energy'] = True

        valid = self['has_aptamer'](sequence)
        valid_spacer = structure[42:-8].find('....') != -1 and structure[42:-8][structure[42:-8].find('....'):].find(
            ')') != -1
        valid_8_U = self['has_8-U'](sequence)
        valid_shape = self['valid_shape'](structure)
        eight_u_unpaired = self['8-U_unpaired'](structure)
        valid_aptamer_hairpin = self['valid_aptamer_hairpin'](structure)

        results['valid_8_U_sequence'] = valid_8_U
        results['valid_spacer_structure'] = valid_spacer
        results['valid_aptamer_sequence'] = valid
        results['valid_shape'] = valid_shape
        results['valid_8-U_structure'] = eight_u_unpaired
        results['valid_aptamer_structure'] = valid_aptamer_hairpin

        if valid and valid_spacer and valid_8_U and valid_shape and eight_u_unpaired and valid_aptamer_hairpin:
            aptamer = sequence[:42]
            eight_u = sequence[-8:]
            spacer, complement = get_spacer_and_complement(sequence[42:-8], structure[42:-8])
            valid_co_fold = self['valid_co_fold'](sequence,
                                                  aptamer,
                                                  spacer,
                                                  structure,
                                                  )
        else:
            valid_co_fold = 0
        results['valid_co_folding'] = valid_co_fold

        results['valid_sequence_and_structure'] = all([
            valid,
            valid_spacer,
            valid_8_U,
            valid_shape,
            valid_co_fold,
            eight_u_unpaired,
            valid_aptamer_hairpin,
        ])

        return results


class BaseDistances(BaseMetrics):
    def __init__(self, metrics):
        super().__init__(metrics)


class NoveltyMetrics(BaseMetrics):
    def __init__(self, metrics):
        super().__init__(metrics)
        self.intersect_size = None
        self.metrics = metrics

    def get_intersect_size(self, generated_df, orig_df, key=None):
        if key is not None:
            gen = generated_df.loc[:, key]
            orig = orig_df.loc[:, key]
        else:
            gen = generated_df if isinstance(generated_df, pd.Series) else generated_df.loc[:, generated_df.columns[0]]
            orig = orig_df if isinstance(generated_df, pd.Series) else orig_df.loc[:, orig_df.columns[0]]

        # check if we have a list that is not a list of chars
        i = orig.index[0]
        if isinstance(orig[i], list) and not isinstance(orig[i][0], str):
            # orig = np.concatenate(orig)
            # gen = np.concatenate(gen)

            # pairs_array_fn = lambda y: np.concatenate(y.apply(lambda x: np.concatenate(np.array(x)), axis=1))
            # length = max(pairs_array_fn(orig).max(), pairs_array_fn(gen).max()) + 1

            orig = orig.apply(lambda x: np.array(x) if len(np.array(x).shape) > 1 else np.array(x))
            gen = gen.apply(lambda x: np.array(x) if len(np.array(x).shape) > 1 else np.array(x))

            # orig_mat = orig.apply(pairs2mat, args= (length, False))
            # gen_mat = gen.apply(pairs2mat, args= (length, False))
            # replace nans with -1 to biased results
            orig = orig.apply(lambda x: np.nan_to_num(x, nan=-1))
            gen = gen.apply(lambda x: np.nan_to_num(x, nan=-1))

            self.intersect_size = gen.apply(lambda x: any(orig.apply(lambda y: np.array_equal(x, y)))).sum()
        else:
            i = orig.index[0]
            if isinstance(orig[i], list) and isinstance(orig[i][0], str):
                orig = orig.apply(lambda x: ''.join(x))
                gen = gen.apply(lambda x: ''.join(x))
            self.intersect_size = gen.isin(orig).sum()
        # print("shared items are ", gen[gen.isin(orig)])
        # print(self.intersect_size)
        return self.intersect_size


class DiversityMetrics(BaseMetrics):
    def __init__(self, metrics):
        super().__init__(metrics)
        self.self_distance = None
        self.metrics = metrics

    def get_self_dist(self, df, distance, key="sequence"):


        if key is not None:
            df = df.loc[:, key]
            if "sequence" in key or "structure" in key:
                ary = cdist(np.asarray(df)[..., None], np.asarray(df)[..., None],
                            metric=lambda x, y: distance(str(x[0]), str(y[0])))

            else:
                # print(df)
                i = df.index[0]
                if isinstance(df[i], list) and all(isinstance(x, (int, float, np.number)) for x in df[i]):
                    df = df.apply(lambda x: np.nanmean(np.array(x)))
                # TODO: There seems to be a bug when there are no pairs, i.e. empty list of pairs.
                # Current workaround: Remove empty lists...
                list_rep = np.asarray([x for x in np.asarray(df)[..., None] if x])
                # ary = cdist(np.asarray(df)[..., None], np.asarray(df)[..., None],
                #             metric=distance)
                # print(list_rep)
                ary = cdist(list_rep, list_rep,
                            metric=distance)
            self.distance = ary
            return self.distance

        df = df.to_numpy()
        ary = cdist(df, df,
                    metric="euclidean")

        self.distance = ary
        return self.distance


class DistributionLearningMetrics(BaseMetrics):
    """
    Evaluated metrics that can be used for distribution-learning design.
    This could be:
      Sequence:
          GC-content of sequence
          General nucleotide contents
          Check amino acids of the sequence? -> design of mRNAs?
      Structure:
          Design for desired substructures
          Design with desired motifs?
          Design for a desired shape
          Riboswitch-like structures
          tRNA-like structures
          General Family like structures
          Design with a certain length?
      match sequences/ structures exactly:
        hamming
        levenshtein
        F1, ...


    """

    def __init__(self,
                 general_dists: List[str],
                 nov_metrics: List[str] = None,
                 div_metrics: List[str] = None,
                 feats=None,
                 feat_spec_dists=None,
                 ):
        """
        :param general_dists: list of distance measures that are used for computing the diversity
        :param nov_metrics: list of novelty metrics that are used for computing the novelty
        :param div_metrics: list of diversity metrics that are used for computing the diversity
        :param feats: list of features that are used for computing the novelty and diversity
        :param feat_spec_dists: dict of lists of distance measures that are used for computing the diversity
        """
        # Novelty metrics takes two datasets and a feature to compute its novelty

        # distance_measures are used to compare between strings and used for computing
        # the diversity
        self.general_distances = {k: get_general_dist(k) for k in general_dists}
        self.general_distances = BaseDistances(self.general_distances)

        if feat_spec_dists is not None:
            self.feat_specific_distances = {}
            for feat, dists in feat_spec_dists.items():
                self.feat_specific_distances[feat] = {dist_name: get_general_dist(dist_name) for dist_name in dists}
                self.feat_specific_distances[feat] = BaseDistances(self.feat_specific_distances[feat])

        if nov_metrics is not None:
            self.novelty_metrics = {k: get_nov_metric(k) for k in nov_metrics}
            self.novelty_metrics = NoveltyMetrics(self.novelty_metrics)
        else:
            self.novelty_metrics = None

        if div_metrics is not None:
            self.diversity_metrics = {k: get_div_metric(k) for k in div_metrics}
            self.diversity_metrics = DiversityMetrics(self.diversity_metrics)
        else:
            self.diversity_metrics = None

        self.kl_div = kl_divergence

    # def __init__(self, metrics, gc=None):
    #     metrics = {k: get_metric(k) for k in metrics}
    #     if gc is not None:
    #         self.gc = gc
    #     super().__init__(metrics)

    def evaluate(self, df):
        dataset_results = {}

        if self.diversity_metrics is not None:
            for feat, distances in self.feat_specific_distances.items():

                for dist_name, dist_fn in distances:
                    if feat in df.columns:
                        curr_self_dist = self.diversity_metrics.get_self_dist(df, dist_fn, feat)
                        for k, m in self.diversity_metrics.metrics.items():
                            # print(f"div_{feat}_{dist_name}_{k}")
                            dataset_results[f"div_{feat}_{dist_name}_{k}"] = m(df, dist_fn, feat,
                                                                               copy.deepcopy(
                                                                                   curr_self_dist))  # TODO remove feat

                    else:
                        print(f"Diversity feature {feat} not in dataset. Skipping...")

        if self.novelty_metrics is not None:
            # results = {}
            # get a dataset with predicted features column replaced by the true features and drop predicted features
            for feat, distances in self.feat_specific_distances.items():

                feat1 = feat if "predicted" not in feat else feat.replace("predicted_", "")
                feat2 = feat if "predicted" in feat else "predicted_" + feat

                if feat1 in df.columns and feat2 in df.columns:


                    # print(df.columns)
                    # print(feat1)
                    # print(feat2)
                    curr_intersect_size = self.novelty_metrics.get_intersect_size(df[[feat2]], df[[feat1]])
                    for k, m in self.novelty_metrics.metrics.items():
                        # print(f"nov_{feat}_{k}")
                        dataset_results[f"nov_{feat}_{k}"] = m(df[[feat2]], df[[feat1]], feat,
                                                               curr_intersect_size)  # TODO remove feat
                else:
                    print(f"Novelty feature {feat} not in dataset. Skipping...")

        return dataset_results


class DistributionLearningComparativeMetrics(BaseMetrics):
    """
    Evaluated metrics that can be used for distribution-learning design.
    This could be:
      Sequence:
          GC-content of sequence
          General nucleotide contents
          Check amino acids of the sequence? -> design of mRNAs?
      Structure:
          Design for desired substructures
          Design with desired motifs?
          Design for a desired shape
          Riboswitch-like structures
          tRNA-like structures
          General Family like structures
          Design with a certain length?
      match sequences/ structures exactly:
        hamming
        levenshtein
        F1, ...


    """

    def __init__(self,
                 general_dists: List[str],
                 nov_metrics: List[str] = None,
                 div_metrics: List[str] = None,
                 feat_spec_dists=None,
                 kl_feats: List[str] = None,
                 ):
        """
        :param general_dists: list of distance measures that are used for computing the diversity
        :param nov_metrics: list of novelty metrics that are used for computing the novelty
        :param div_metrics: list of diversity metrics that are used for computing the diversity
        :param feat_spec_dists: dict of lists of distance measures that are used for computing the diversity
        """
        # Novelty metrics takes two datasets and a feature to compute its novelty

        # distance_measures are used to compare between strings and used for computing
        # the diversity
        self.general_distances = {k: get_general_dist(k) for k in general_dists}
        self.general_distances = BaseDistances(self.general_distances)

        if feat_spec_dists is not None:
            self.feat_specific_distances = {}
            for feat, dists in feat_spec_dists.items():
                self.feat_specific_distances[feat] = {dist_name: get_general_dist(dist_name) for dist_name in dists}
                self.feat_specific_distances[feat] = BaseDistances(self.feat_specific_distances[feat])

        if nov_metrics is not None:
            self.novelty_metrics = {k: get_nov_metric(k) for k in nov_metrics}
            self.novelty_metrics = NoveltyMetrics(self.novelty_metrics)
        else:
            self.novelty_metrics = None

        if div_metrics is not None:
            self.diversity_metrics = {k: get_div_metric(k) for k in div_metrics}
            self.diversity_metrics = DiversityMetrics(self.diversity_metrics)
        else:
            self.diversity_metrics = None
        self.kl_feats = kl_feats

    # def __init__(self, metrics, gc=None):
    #     metrics = {k: get_metric(k) for k in metrics}
    #     if gc is not None:
    #         self.gc = gc
    #     super().__init__(metrics)

    def evaluate(self, orig_df, gen_df):
        dataset_results = {}
        dist_params_orig = {}
        dist_params_gen = {}

        if self.diversity_metrics is not None:
            for feat, distances in self.feat_specific_distances.items():
                if feat in gen_df.columns:

                    for dist_name, dist_fn in distances:
                        curr_self_dist = self.diversity_metrics.get_self_dist(gen_df, dist_fn, feat)
                        for k, m in self.diversity_metrics.metrics.items():
                            # print(f"div_{feat}_{dist_name}_{k}")
                            dataset_results[f"div_{feat}_{dist_name}_{k}"] = m(gen_df, dist_fn, feat,
                                                                               copy.deepcopy(
                                                                                   curr_self_dist))  # TODO remove feat
                else:
                    print(f"Diversity feature {feat} not in dataset. Skipping...")


        if self.novelty_metrics is not None:
            # results = {}
            # get a dataset with predicted features column replaced by the true features and drop predicted features
            for feat, _ in self.feat_specific_distances.items():


                feat = feat if "predicted" not in feat else feat.replace("predicted_", "")

                if feat in orig_df.columns and feat in gen_df.columns:

                    curr_intersect_size = self.novelty_metrics.get_intersect_size(gen_df[[feat]], orig_df[[feat]])
                    for k, m in self.novelty_metrics.metrics.items():
                        # print(f"nov_{feat}_{k}")
                        dataset_results[f"nov_{feat}_{k}"] = m(orig_df[[feat]], gen_df[[feat]], feat,
                                                               curr_intersect_size)  # TODO remove feat
                else:
                    print(f"Novelity feature {feat} not in dataset. Skipping...")

        if self.kl_feats is not None:
            for feat in self.kl_feats:
                dtype = orig_df[feat].dtype
                if feat in orig_df.columns and feat in gen_df.columns:
                    if dtype == bool:
                        dist_params_orig[feat] = Bernoulli(probs=orig_df[feat].mean())
                        dist_params_gen[feat] = Bernoulli(probs=gen_df[feat].mean())
                    elif dtype == float or dtype == int:
                        dist_params_orig[feat] = Normal(loc=orig_df[feat].mean(), scale=orig_df[feat].std() + 1e-5)
                        dist_params_gen[feat] = Normal(loc=gen_df[feat].mean(), scale=gen_df[feat].std() + 1e-5)
                    elif dtype == object:
                        dist_params_orig[f"{feat}_len"] = Normal(loc=orig_df[feat].apply(len).mean(),
                                                                 scale=orig_df[feat].apply(len).std() + 1e-5)
                        dist_params_orig[f"{feat}_mean"] = Normal(loc=orig_df[feat].apply(np.mean).mean(),
                                                                  scale=orig_df[feat].apply(np.mean).std() + 1e-5)
                        dist_params_gen[f"{feat}_len"] = Normal(loc=gen_df[feat].apply(len).mean(),
                                                                scale=gen_df[feat].apply(len).std() + 1e-5)
                        dist_params_gen[f"{feat}_mean"] = Normal(loc=gen_df[feat].apply(np.mean).mean(),
                                                                 scale=gen_df[feat].apply(np.mean).std() + 1e-5)
                    else:
                        raise ValueError(f"Feature {feat} has unsupported type {dtype}")
                    # check type of feature
                else:
                    print(f"KL feature {feat} not in dataset. Skipping...")

        # add kl divergence to dataset results
        for key in dist_params_orig.keys():
            dataset_results[f"kl_{key}"] = kl_divergence(dist_params_orig[key], dist_params_gen[key]).item()

        return dataset_results


################################################################################
# Functions
################################################################################

################################################################################
# For Strings
################################################################################

################################################################################
# Structure
################################################################################

# is tested
# TODO: implement all metrics for the case that we have
def abs_pair_ratio_dif_from_string(s1, true_pairs):
    """
    Returns the absolute difference of the pair ratio of a predicted structure
    in dot-bracket format compared to a ground truth list of pairs with the same
    length.
    """
    # pair_ratio_1 = s1.count('(') +  / len(s1))
    pp = np.asarray(db2pairs(s1))
    tp = np.asarray(true_pairs)
    pair_ratio_1 = len(pp.flatten()) / len(s1)
    pair_ratio_2 = len(tp.flatten()) / len(s1)
    # print(pair_ratio_2)
    # print(pair_ratio_1)
    return np.abs(pair_ratio_1 - pair_ratio_2)

# is tested
def abs_pair_ratio_score(pred, true_pairs):
    """
    Returns a score (1-loss) of the absolute pair ratio diff.
    """
    return to_score(abs_pair_ratio_dif_from_string(pred, true_pairs))

# is tested
# hamming distance is kind of incorrect when using multiplets, this is a bad metric then
def hamming_distance(s1, true_pairs, s2=None,
                     len_mismatch_nan=True):  # TODO: include hamming distance when using 'don't care' symbol
    if s2 is None:
        s2 = pairs2db(true_pairs)
    if len(s1) != len(s2):
        return len(s2) if not len_mismatch_nan else np.nan
    l1 = np.asarray(list(s1))
    l2 = np.asarray(list(s2))

    distance = np.sum((l1 != l2).astype(np.int8))
    return distance

# is tested
def rel_hamming_list_score(l1, l2):
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)

    distance = np.sum((l1 != l2).astype(np.int8))

    return 1 - (distance / len(l1))

# is tested
def hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=False):
    return hamming_distance(s1, true_pairs, s2=s2, len_mismatch_nan=len_mismatch_nan) / len(s2)

# is tested
def hamming_distance_score(pred, true_pairs):
    return to_score(hamming_distance_relative(pred, true_pairs))


# is tested
def f1_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs, length=len(pred))
    return f1_score_from_matrices(pred_mat, true_mat)


def mcc_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return mcc_from_matrices(pred_mat, true_mat)


def recall_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return recall_score_from_matrices(pred_mat, true_mat)


def specificity_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return specificity_score_from_matrices(pred_mat, true_mat)


def precision_score_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return precision_score_from_matrices(pred_mat, true_mat)


def solved_from_string(pred, true):
    return int(pred == true)


def non_correct_from_string(pred, true_pairs):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    tp = tp_from_matrices(pred_mat, true_mat)
    return non_correct(tp)


def weisfeiler_lehman_score_from_string(pred, true_pairs, kernel='WeisfeilerLehman'):
    pred_mat = db2mat(pred)
    true_mat = pairs2mat(true_pairs)
    return graph_distance_score_from_matrices(pred_mat, true_mat, kernel=kernel)


################################################################################
# from e2efold
################################################################################
# we first apply a kernel to the ground truth a
# then we multiple the kernel with the prediction, to get the TP allows shift
# then we compute f1
# we unify the input all as the symmetric matrix with 0 and 1, 1 represents pair
def evaluate_shifted_f1(pred_a, true_a):
    pred_a = torch.tensor(pred_a)
    true_a = torch.tensor(true_a)

    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, 1.0, 1.0],
                       [0.0, 1.0, 0.0]])
    pred_a_filtered = signal.convolve2d(pred_a, kernel, 'same')
    fn = len(torch.where((true_a - torch.Tensor(pred_a_filtered)) == 1)[0])
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    tp = true_p - fn
    fp = pred_p - tp
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    return f1_score.item()


################################################################################
################################################################################


def eval_metrics(pred, true, sequence, pk_pairs, metrics):
    results = {}
    # results['hamming'] = hamming_distance(pred, true_pairs)
    # results['abs_pair_ratio_dif'] = abs_pair_ratio_dif_from_string(pred, true_pairs)
    if pred is None or isinstance(pred, float) or len(pred) == 0:
        pred_mat = np.zeros((len(sequence), len(sequence)))  # TODO: return 0 for all metrics
    elif isinstance(pred, str):
        pred_mat = db2mat(pred)
    elif isinstance(pred, list):
        if isinstance(pred[0], str):
            pred_mat = db2mat(''.join(str))
        elif isinstance(pred[0], tuple) or isinstance(pred[0], list):
            if len(pred[0]) == 2:
                pred_mat = pairs2mat(pred, length=len(sequence), no_pk=True)
            else:
                pred_mat = pairs2mat(pred, length=len(sequence))
    elif isinstance(pred, np.ndarray):  # is matrix already
        pred_mat = pred
    # metrics = {}
    if true is None or isinstance(true, float) or len(true) == 0:
        true_mat = np.zeros((len(sequence), len(sequence)))  # TODO: This should actually raise an exception!
    elif isinstance(true, str):
        true_mat = db2mat(true)
    elif isinstance(true, list):
        if isinstance(true[0], str):
            true_mat = db2mat(''.join(true))
        elif isinstance(true[0], tuple) or isinstance(true[0], list):
            true_mat = pairs2mat(true, length=len(sequence))
    elif isinstance(true, np.ndarray):  # is matrix already
        true_mat = true

    vocab = list(set(sequence))
    pair_matrices = get_pair_specific_matrices(pred_mat, true_mat, sequence, pk_pairs, vocab)

    node_labels = {i: s for i, s in enumerate(sequence)}

    for pair_type, (pred_mat, true_mat) in pair_matrices.items():

        pair_type = '_' + pair_type
        results['weisfeiler_lehman' + pair_type] = graph_distance_score_from_matrices(
            pred_mat,
            true_mat,
            kernel='WeisfeilerLehman'
        )

        results['weisfeiler_lehman_with_seq' + pair_type] = graph_distance_score_from_matrices(
            pred_mat,
            true_mat,
            kernel='WeisfeilerLehman',
            node_labels=node_labels,
        )

        results['shifted_f1' + pair_type] = evaluate_shifted_f1(torch.tensor(pred_mat), torch.tensor(true_mat))

        results['solved' + pair_type] = solved_from_mat(pred_mat, true_mat)

        tp = tp_from_matrices(pred_mat, true_mat)
        fp = get_fp(pred_mat, tp)
        fn = get_fn(true_mat, tp)
        tn = tn_from_matrices(pred_mat, true_mat)
        results.update({name + pair_type: metric(tp, fp, tn, fn) for name, metric in metrics})
    return results


def get_pair_specific_matrices(pred_mat, true_mat, seq, pk_pairs, vocab):
    watson_crick_pairs = ['GC', 'CG', 'AU', 'UA']
    canonical_pairs = watson_crick_pairs + ['GU', 'UG']

    pair_matrices = {''.join(sorted(x)): [np.zeros((len(seq), len(seq))), np.zeros((len(seq), len(seq)))] for x in
                     itertools.product(vocab, repeat=2)}

    pair_matrices['watson_crick_pairs'] = [np.zeros((len(seq), len(seq))), np.zeros((len(seq), len(seq)))]
    pair_matrices['canonical_pairs'] = [np.zeros((len(seq), len(seq))), np.zeros((len(seq), len(seq)))]
    pair_matrices['non-canonical_pairs'] = [np.zeros((len(seq), len(seq))), np.zeros((len(seq), len(seq)))]
    pair_matrices['multiplets_pairs'] = [np.zeros((len(seq), len(seq))), np.zeros((len(seq), len(seq)))]
    pair_matrices['pseudoknot_pairs'] = [np.zeros((len(seq), len(seq))), np.zeros((len(seq), len(seq)))]

    for pair in pk_pairs:
        pair_matrices['pseudoknot_pairs'][1][pair[0], pair[1]] = 1
        pair_matrices['pseudoknot_pairs'][1][pair[1], pair[0]] = 1

    pair_matrices['pseudoknot_pairs'][0] = pair_matrices['pseudoknot_pairs'][1] * pred_mat

    for pred_pair, true_pair in itertools.zip_longest(np.argwhere(pred_mat),
                                                      np.argwhere(true_mat),
                                                      fillvalue=None):

        if pred_pair is not None:
            pair_type = ''.join(sorted(seq[pred_pair[0]] + seq[pred_pair[1]]))
            pair_matrices[pair_type][0][pred_pair[0], pred_pair[1]] = 1
            # pair_matrices[pair_type][0][pred_pair[1], pred_pair[0]] = 1

            if pair_type in watson_crick_pairs:
                pair_matrices['watson_crick_pairs'][0][pred_pair[0], pred_pair[1]] = 1
                pair_matrices['canonical_pairs'][0][pred_pair[0], pred_pair[1]] = 1
                # pair_matrices['watson_crick_pairs'][0][pred_pair[1], pred_pair[0]] = 1
            elif pair_type in canonical_pairs:
                pair_matrices['canonical_pairs'][0][pred_pair[0], pred_pair[1]] = 1
                # pair_matrices['canonical_pairs'][0][pred_pair[1], pred_pair[0]] = 1
            else:
                pair_matrices['non-canonical_pairs'][0][pred_pair[0], pred_pair[1]] = 1
                # pair_matrices['non-canonical_pairs'][0][pred_pair[1], pred_pair[0]] = 1
            if pred_mat[pred_pair[0], :].sum() > 1:
                pair_matrices['multiplets_pairs'][0][pred_pair[0], pred_pair[1]] = 1
                # pair_matrices['multiplets_pairs'][0][pred_pair[1], pred_pair[0]] = 1

        if true_pair is not None:
            pair_type = ''.join(sorted(seq[true_pair[0]] + seq[true_pair[1]]))
            pair_matrices[pair_type][1][true_pair[0], true_pair[1]] = 1
            # pair_matrices[pair_type][1][true_pair[1], true_pair[0]] = 1

            if pair_type in watson_crick_pairs:
                pair_matrices['watson_crick_pairs'][1][true_pair[0], true_pair[1]] = 1
                # pair_matrices['watson_crick_pairs'][1][true_pair[1], true_pair[0]] = 1
            elif pair_type in canonical_pairs:
                pair_matrices['canonical_pairs'][1][true_pair[0], true_pair[1]] = 1
                # pair_matrices['canonical_pairs'][1][true_pair[1], true_pair[0]] = 1
            else:
                pair_matrices['non-canonical_pairs'][1][true_pair[0], true_pair[1]] = 1
                # pair_matrices['non-canonical_pairs'][1][true_pair[1], true_pair[0]] = 1
            if true_mat[true_pair[0], :].sum() > 1:
                pair_matrices['multiplets_pairs'][1][true_pair[0], true_pair[1]] = 1
                # pair_matrices['multiplets_pairs'][1][true_pair[1], true_pair[0]] = 1

    pair_matrices.update({'all': [pred_mat, true_mat]})
    return pair_matrices


def get_pair_positions(mat):
    return np.argwhere(mat)


#     metrics['f1'] = f1(tp, fp, fn)
#     metrics['mcc'] = mcc(tp, tn, fp, fn)
#     metrics['precision'] = precision(tp, fp)
#     metrics['recall'] = recall(tp, fn)
#     metrics['specificity'] = specificity(tn, fp)
#
#     metrics['tp'] = tp
#     metrics['tn'] = tn
#     metrics['fp'] = fp
#     metrics['fn'] = fn
#
#
#     return metrics


################################################################################
# Sequence
################################################################################


def gc_content_abs(s1, s2):
    gc1 = (s1.upper().count('G') + s1.upper().count('C')) / len(s1)
    gc2 = (s2.upper().count('G') + s2.upper().count('C')) / len(s2)
    return abs(gc1 - gc2)


def gc_content_abs_score(pred, true):
    return to_score(gc_content_abs(pred, true))


################################################################################
# For Matrices
################################################################################

################################################################################
# Structure
################################################################################

def f1_score_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    fn = get_fn(true, tp)
    return f1(tp, fp, None, fn)


def mcc_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    fn = get_fn(true, tp)
    tn = tn_from_matrices(pred, true)
    return mcc(tp, tn, fp, fn)


def recall_score_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fn = get_fn(true, tp)
    return recall(tp, fn)


def specificity_score_from_matrices(pred, true):
    tn = tn_from_matrices(pred, true)
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    return specificity(tn, fp)


def precision_score_from_matrices(pred, true):
    tp = tp_from_matrices(pred, true)
    fp = get_fp(pred, tp)
    return precision(tp, fp)


def solved_from_mat(pred, true):
    solved = np.all(np.equal(true, pred)).astype(int)
    return solved


def graph_distance_score_from_pairs(pred, true, kernel='WeisfeilerLehman', node_labels=None):
    if isinstance(true[0], list):
        true = np.concatenate(true)
        pred = np.concatenate(pred)
    length = max(true.max(), pred.max()) + 1
    true_mat = pairs2mat(true, length=length, no_pk=False)
    pred_mat = pairs2mat(pred, length=length, no_pk=False)
    return graph_distance_score_from_matrices(pred_mat, true_mat, kernel, node_labels=node_labels)


def graph_distance_score_from_matrices(pred, true, kernel, node_labels=None):
    pred_graph = mat2graph(pred, node_labels=node_labels)
    true_graph = mat2graph(true, node_labels=node_labels)
    kernel = get_graph_kernel(kernel=kernel)
    kernel.fit_transform([true_graph])
    distance_score = kernel.transform([pred_graph])  # TODO: Check output, might be list or list of lists

    return distance_score[0][0]


################################################################################
# Sequence
################################################################################


################################################################################
# Helpers
################################################################################

def get_graph_kernel(kernel, n_iter=5, normalize=True):
    if kernel == 'WeisfeilerLehman':
        return WeisfeilerLehman(n_iter=n_iter,
                                normalize=normalize,
                                base_graph_kernel=VertexHistogram)
    elif kernel == 'WeisfeilerLehmanOptimalAssignment':
        return WeisfeilerLehmanOptimalAssignment(n_iter=n_iter,
                                                 normalize=normalize)
    elif kernel == 'ShortestPath':
        return ShortestPath(normalize=normalize)


def mat2graph(matrix, node_labels=None):
    if node_labels is not None:
        graph = Graph(initialization_object=matrix.astype(int),
                      node_labels=node_labels)  # TODO: Think about if we need to label the nodes differenty
    else:
        graph = Graph(initialization_object=matrix.astype(int),
                      node_labels={s: str(s) for s in
                                   range(
                                       matrix.shape[0])})  # TODO: Think about if we need to label the nodes differenty

    return graph

# is tested
def f1(tp, fp, tn, fn):
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1_score

# is tested
def recall(tp, fp, tn, fn):
    recall = tp / (tp + fn + 1e-8)
    return recall

# is tested
def specificity(tp, fp, tn, fn):
    specificity = tn / (tn + fp + 1e-8)
    return specificity

# is tested
def precision(tp, fp, tn, fn):
    precision = tp / (tp + fp + 1e-8)
    return precision

# is tested
def mcc(tp, fp, tn, fn):
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    return mcc


def non_correct(tp, fp, tn, fn):
    non_correct = (tp == 0).astype(int)
    return non_correct

# is tested
def tp_from_matrices(pred, true):
    tp = np.logical_and(pred, true).sum()
    return tp

# is tested
def tn_from_matrices(pred, true):
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true)).sum()
    return tn

# is tested
def get_fp(pred, tp):
    fp = pred.sum() - tp
    return fp

# is tested
def get_fn(true, tp):
    fn = true.sum() - tp
    return fn


def to_score(metric):
    return 1 - metric


################################################################################
# Diversity
################################################################################


def self_diff(df, distance, key="sequence"):
    if key is not None:
        df = df.loc[:, key]
        if key == "sequence" or key == "structure":
            ary = cdist(np.asarray(df)[..., None], np.asarray(df)[..., None],
                        metric=lambda x, y: distance(str(x[0]), str(y[0])))
        else:
            ary = cdist(np.asarray(df)[..., None], np.asarray(df)[..., None],
                        metric=distance)
        return ary
        df = df.to_numpy()
    ary = cdist(df, df,
                metric="euclidean")
    return ary


def diameter(df, distance, key="sequence", self_diff_mat=None):  # (df, distance, key="sequence"):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    return np.nanmax(self_diff_mat)


def diversity(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    np.fill_diagonal(self_diff_mat, np.nan)
    return np.nanmean(self_diff_mat)


def bottleneck(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    np.fill_diagonal(self_diff_mat, np.nan)
    return np.nanmin(self_diff_mat)


def sum_bottleneck(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    np.fill_diagonal(self_diff_mat, np.nan)
    return np.sum(np.nanmin(self_diff_mat, -1))


def DPP(df, distance, key="sequence", self_diff_mat=None):
    if self_diff_mat is None:
        self_diff_mat = self_diff(df, distance, key)
    return np.linalg.det(1 - self_diff_mat)


def get_div_metric(m):
    if m == 'DPP':
        return DPP
    elif m == 'sum_btlnk':
        return sum_bottleneck
    elif m == 'diameter':
        return diameter
    elif m == 'diversity':
        return diversity


################################################################################
# Novelty
################################################################################
def intersect_size(orig_df, generated_df, key="sequence"):
    gen = generated_df.loc[:, key]
    orig = orig_df.loc[:, key]
    print("shared items are ", gen[gen.isin(orig), key])
    return gen.isin(orig).sum()


def novelty(orig_df, generated_df, key="sequence", inter_size=None):
    gen_len = generated_df.shape[0]
    if inter_size is None:
        gen = generated_df.loc[:, key]
        orig = orig_df.loc[:, key]
        inter_size = intersect_size(gen, orig)
    return (1 - inter_size / gen_len)  # .values[0]


def iou(orig_df, generated_df, key="sequence", inter_size=None):
    gen_len = generated_df.shape[0]
    orig_len = orig_df.shape[0]
    if inter_size is None:
        gen = generated_df.loc[:, key]
        orig = orig_df.loc[:, key]
        inter_size = intersect_size(gen, orig)
    union_size = gen_len + orig_len - inter_size
    return (1 - inter_size / union_size) if union_size != 0 else 0


def get_nov_metric(m):
    if m == 'iou':
        return iou
    elif m == 'novelty':
        return novelty



################################################################################
# Riboswitch Metrics
################################################################################

# is tested
def eval_shape(dot_bracket):
    return RNA.abstract_shapes(dot_bracket) == '[][]'

# is tested
def check_aptamer_structure(structure, len_aptamer=42):
    return '(((((.....)))))' in structure[:len_aptamer]

# is tested
def check_8_u(structure, length=7):
    return structure[-length:] == '.' * length

# is tested
def evaluate_co_transcriptional_folding_simulation(sequence,
                                                   aptamer,
                                                   spacer,
                                                   structure,
                                                   elongation=10):
    """
    Form the paper:
    In the current implementation, folding paths are represented as a sequence
    of secondary structures computed for sub-sequences starting at the 5′-end.
    We used individual transcription steps of 5–10 nt to simulate
    co-transcriptional folding with varying elongation speed. Secondary
    structures are computed by RNAfold, a component of the Vienna RNA Package
    (22), with parameter settings -d2 -noPS -noLP. If one of the transcription
    intermediates forms base pairs between aptamer and spacer, it is likely
    that this will interfere with the ligand-binding properties;
    hence, such a candidate is rejected.

    Changes:
    - just calling RNA.fold instead of RNAFold -d1 -noPS -noLP
    - fixed elongation speed of 10 to save compute
    """

    aptamer_len = len(aptamer)
    spacer_len = len(spacer)

    valid_intermediates = []

    for i in range(1, len(sequence), elongation):
        struc, energy = RNA.fold(sequence[:i])
        seq = sequence[:i]

        pairs_dict = db2pairs_dict_closers_keys(struc)

        if not spacer in seq:
            continue
        else:
            spacer_idx = seq.index(spacer)
            assert seq[spacer_idx:spacer_idx + spacer_len] == spacer
            if not ')' in struc[spacer_idx:spacer_idx + spacer_len]:
                valid_intermediates.append(True)
                continue
            else:
                spacer_pair_ids = []
                for i, s in enumerate(struc[spacer_idx:spacer_idx + spacer_len], spacer_idx):
                    if s == ')':
                        if pairs_dict[i] < aptamer_len - 1:
                            return False
                    valid_intermediates.append(True)
    return all(valid_intermediates)


def db2pairs_dict_closers_keys(structure):
    stack = []
    pairs = []
    for i, s in enumerate(structure):
        if s == '.':
            continue
        elif s == '(':
            stack.append(i)
        else:
            pairs.append([stack.pop(), i])
    return {p2: p1 for p1, p2 in pairs}

# is tested
def has_aptamer(seq):
    return seq[:42] == 'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA'

# is tested
def has_8_U(seq):
    return seq[-8:] == 'UUUUUUUU'

# is tested
def get_spacer_and_complement(seq, struc):
    """
    get the spacer and complement

    the spacer
    """
    stack = []
    for i, sym in enumerate(struc):
        if sym == '.':
            continue
        elif sym == '(':
            stack.append(i)
        else:
            if not stack:
                return seq[:i], seq[i:]
            stack.pop()
    return seq[:i], seq[i:]
