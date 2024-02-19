import pytest
import numpy as np

from RnaBench.lib.utils import db2pairs, pairs2mat
from RnaBench.lib.metrics import GoalDirectedMetrics, DistributionLearningMetrics
from RnaBench.lib.metrics import (
abs_pair_ratio_dif_from_string,
abs_pair_ratio_score,
hamming_distance,
rel_hamming_list_score,
hamming_distance_relative,
f1_score_from_string,
tp_from_matrices,
tn_from_matrices,
get_fp,
get_fn,
f1,
recall,
specificity,
precision,
mcc,
eval_shape,
check_aptamer_structure,
check_8_u,
evaluate_co_transcriptional_folding_simulation,
has_aptamer,
has_8_U,
get_spacer_and_complement,
)


test_sequence = 'GCACUUUUGUGC'
test_structure = '((((....))))'
test_pairs = [[0, 11, 0], [1, 10, 0], [2, 9, 0], [3, 8, 0]]

@pytest.mark.parametrize("metrics", [['f1_score',
                                      'mcc',
                                      'f1_shifted',
                                      'wl',
                                      'recall',
                                      'precision',
                                      'specificity',
                                      'solved'], ['f1_score']])
def test_goaldirected_metrics_general(metrics):
    gm = GoalDirectedMetrics(metrics)


@pytest.mark.parametrize("nov_metrics", [["iou", "novelty"], None])
@pytest.mark.parametrize("div_metrics", [["diversity", "diameter", "DPP", "sum_btlnk"], None])
@pytest.mark.parametrize("gen_dists", [["hamming", "lv"]])
@pytest.mark.parametrize("feats", [["sequence", "structure"], None])
@pytest.mark.parametrize("feat_distmap",[{"predicted_sequence": ["hamming", "lv"],
                                          "predicted_s_0" : ["l2"],
                                          "predicted_h_0" : ["l2"],
                                          "predicted_i_0" : ["l2"],
                                          "predicted_i_1": ["l2"],
                                          "predicted_pairs": ["wl_pairs"]}, None])
def test_dist_learning_metrics_general(nov_metrics,
                                       div_metrics,
                                       gen_dists,
                                       feats,
                                       feat_distmap,
                                       ):
    dm = DistributionLearningMetrics(general_dists=gen_dists,
                                     nov_metrics=nov_metrics,
                                     div_metrics=div_metrics,
                                     feats=feats,
                                     feat_spec_dists=feat_distmap,
                                    )


def test_abs_pair_ratio_dif_from_string():
    res = abs_pair_ratio_dif_from_string(test_structure, test_pairs)
    assert 0.0 == res


def test_abs_pair_ratio_score():
    res = abs_pair_ratio_score(test_structure, test_pairs)
    assert 1.0 == res


def test_hamming_distance():
    s1 = test_structure
    s2 = None
    true_pairs = test_pairs

    assert 0 == hamming_distance(s1, true_pairs, s2=s2)

    s1 = s1[1:-1]
    assert np.isnan(hamming_distance(s1, true_pairs, s2=s2))

    s2 = test_structure
    assert len(s2) == hamming_distance(s1, true_pairs, s2=s2, len_mismatch_nan=False)

    s1 = test_structure
    true_pairs = None
    assert 0 == hamming_distance(s1, true_pairs, s2=s2)

    s1 = '(((......)))'

    assert 2 == hamming_distance(s1, true_pairs, s2=s2)


@pytest.mark.parametrize(
    ("s1", "s2", "expected"),
    [
        (test_structure, test_structure, 1.0),
        pytest.param(test_structure[1:-1], test_structure, 1.0, marks=pytest.mark.xfail(reason='Length mismatch')),
        (test_structure, '....(())....', 0.0),
    ],
)
def test_rel_hamming_list_score(s1, s2, expected):

    assert expected == rel_hamming_list_score(list(s1), list(s2))


def test_hamming_distance_relative():
    s1 = test_structure
    s2 = test_structure
    true_pairs = test_pairs
    len_mis = True

    assert 0.0 == hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)

    s1 = '....(())....'
    s2 = test_structure
    true_pairs = test_pairs
    len_mis = True

    assert 1.0 == hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)

    s1 = '(((......)))'
    s2 = test_structure
    true_pairs = test_pairs
    len_mis = True

    assert (2 / len(s1)) == hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)


@pytest.mark.xfail
def test_hamming_distance_relative():
    s1 = test_structure
    s2 = None
    true_pairs = test_pairs
    len_mis = True

    hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)

    s1 = None
    s2 = test_structure
    true_pairs = test_pairs
    len_mis = True

    hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)

    s1 = test_structure[1:-1]
    s2 = test_structure
    true_pairs = test_pairs
    len_mis = True

    hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)

    s1 = test_structure[1:-1]
    s2 = test_structure
    true_pairs = test_pairs
    len_mis = False

    hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)

    s1 = test_structure
    s2 = None
    true_pairs = None
    len_mis = True

    hamming_distance_relative(s1, s2, true_pairs, len_mismatch_nan=len_mis)


def test_f1_score_from_string():
    s1 = test_structure
    true_pairs = test_pairs
    assert 1.0 == pytest.approx(f1_score_from_string(s1, true_pairs))

def test_tp_from_matrices():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    assert 8 == tp_from_matrices(pred_mat, true_mat)

def test_tn_from_matrices():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    assert 136 == tn_from_matrices(pred_mat, true_mat)

def test_get_fn():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    assert 0 == get_fn(true_mat, tp)

def test_get_fp():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    assert 0 == get_fp(pred_mat, tp)


def test_f1():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    tn = tn_from_matrices(pred_mat, true_mat)
    fn = get_fn(true_mat, tp)
    fp = get_fp(pred_mat, tp)
    assert 1.0 == pytest.approx(f1(tp, fp, tn, fn))

def test_recall():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    tn = tn_from_matrices(pred_mat, true_mat)
    fn = get_fn(true_mat, tp)
    fp = get_fp(pred_mat, tp)
    assert 1.0 == pytest.approx(recall(tp, fp, tn, fn))

def test_specificity():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    tn = tn_from_matrices(pred_mat, true_mat)
    fn = get_fn(true_mat, tp)
    fp = get_fp(pred_mat, tp)
    assert 1.0 == pytest.approx(specificity(tp, fp, tn, fn))

def test_precision():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    tn = tn_from_matrices(pred_mat, true_mat)
    fn = get_fn(true_mat, tp)
    fp = get_fp(pred_mat, tp)
    assert 1.0 == pytest.approx(precision(tp, fp, tn, fn))

def test_mcc():
    pred_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    true_mat = pairs2mat(db2pairs(test_structure), length=len(test_structure))
    tp = tp_from_matrices(pred_mat, true_mat)
    tn = tn_from_matrices(pred_mat, true_mat)
    fn = get_fn(true_mat, tp)
    fp = get_fp(pred_mat, tp)
    assert 1.0 == pytest.approx(mcc(tp, fp, tn, fn))

@pytest.mark.parametrize(("structure", "expected"),
                          [
                          ('...((((...))))....', False),
                          ('..(((..)))..((((....))))', True),
                          ('...(.).(.)..', True),
                          ('..(.).(.).(.)', False),
                          pytest.param('..........', False, marks=pytest.mark.xfail(reason='Internal error in site package RNA')),
                          ],
                          )
def test_eval_shape(structure, expected):
    assert eval_shape(structure) == expected

@pytest.mark.parametrize(("structure", "expected"),
                          [
                          ('(((((.....)))))', True),
                          ('......(((((.....)))))....', True),
                          ('.(((...)))(((((.....)))))....', True),
                          ('........................................................(((...)))(((((.....)))))', False),
                          ],
                          )
def test_check_aptamer_structure(structure, expected):
    assert expected == check_aptamer_structure(structure)

@pytest.mark.parametrize(("structure", "expected"),
                          [
                          ('........', True),
                          ('......(((((.....)))))....', False),
                          ('.(((...)))(((((.....)))))...', False),
                          ('...((((....)))).......', True),
                          ('...((((....))))...........', True),
                          ],
                          )
def test_check_8_u(structure, expected):
    assert expected == check_8_u(structure)


@pytest.mark.parametrize(("sequence", "aptamer", "spacer", "structure", "expected"),
                          [
                          ('AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAUUACAUCUGAAGUGCUGCCUUUUUUUU',
                           'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                           'UUACAUC',
                           '...........(((((.....)))))....((((((((((((.......))))))))))))........',
                           True,
                           ),  # RS1
                           ('AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAUGAUCUCGCUUGAAGUGCUGCUUUUUUUU',
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                            'UGAUCUCGCU',
                            '...........(((((.....)))))....((((((((((((..........)))))))))))).......',
                            True,
                            ),  # RS2
                            (
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAUUUACAUACUCGGUAAACUGAAGUGCUGCCAUUUUUUUU',
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                            'UUUACAUACUCGGUAAAC',
                            '...........(((((.....)))))...((((((((((((((((((.......))))).)))))))))))))........',
                            True,
                            ),  # RS3
                            (
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAAACCGAAAUUUGCGCUUGAAGUGCUGCUUUUUUUU',
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                            'AACCGAAAUUUGCGCU',
                            '...........(((((.....)))))....(((((((((((((..((.......)).))))))))))))).......',
                            True,
                            ),  # RS4
                            (
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCACUCCUAGUGGAGUGAAGUGCUGUUUUUUUU',
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                            'CUCCUAGUGGAG',
                            '........((((((((.....)))))...)))((((((((((((((....))))))))))))))........',
                            True,
                            ),  # RS8
                            (
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAGAAAUCUCUGAAGUGCUGUUUUUUUU',
                            'AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                            'GAAAUCUC',
                            '........((((((((.....)))))...)))((((((((((((....))))))))))))........',
                            True,
                            ),  # RS10
                          ],
                          )
def test_evaluate_co_transcriptional_folding_simulation(
                                                        sequence,
                                                        aptamer,
                                                        spacer,
                                                        structure,
                                                        expected,
                                                        ):
    assert expected == evaluate_co_transcriptional_folding_simulation(
                                                                      sequence,
                                                                      aptamer,
                                                                      spacer,
                                                                      structure,
                                                                      )

@pytest.mark.parametrize(("sequence", "expected"),
                         [
                         ('AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAGAAAUCUCUGAAGUGCUGUUUUUUUU', True),
                         ('GUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAGAAAUCUCUGAAGUGCUGUUUUUUUU', False),
                         ])
def test_has_aptamer(sequence, expected):
    assert expected == has_aptamer(sequence)


@pytest.mark.parametrize(("sequence", "expected"),
                         [
                         ('AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAGAAAUCUCUGAAGUGCUGUUUUUUUU', True),
                         ('GUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAGAAAUCUCUGAAGUGCUGUUUUUU', False),
                         ])
def test_has_8_U(sequence, expected):
    assert expected == has_8_U(sequence)

@pytest.mark.parametrize(("seq", "struc", "exp_spacer", "exp_comp"),
                         [
                         (
                         'GAAAUCUCUGAAGUGCUG',
                         '((....))))))))))))',
                         'GAAAUCUC',
                         'UGAAGUGCUG',
                         ),
                         (
                         'CGUCAGAAAUCUCUGAAGUGCUG',
                         '((((.......)))))))))...',
                         'CGUCAGAAAUCUCUG',
                         'AAGUGCUG',
                         ),
                         ])
def test_get_spacer_and_complement(seq, struc, exp_spacer, exp_comp):
    spacer, complement = get_spacer_and_complement(seq, struc)
    assert spacer == exp_spacer
    assert complement == exp_comp