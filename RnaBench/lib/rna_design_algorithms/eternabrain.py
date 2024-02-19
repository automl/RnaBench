

import sys
from argparse import ArgumentParser
import tensorflow as tf
import os
import pickle
from typing import Optional
import numpy as np
import RNA
import copy
from numpy.random import choice
from difflib import SequenceMatcher

from EternaBrain.rna_prediction.readData import format_pairmap
from EternaBrain.rna_prediction.sap1 import sbc
from EternaBrain.rna_prediction.sap2 import dsp
import time

def save_timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        out_dir = kwargs.get('out_dir', '')
        #check if path exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        result_str = f"{result[0]}, {result[1]}, {duration}"
        with open(f'{out_dir}/timing_{kwargs["index"]}.txt', 'w') as f:
            f.write(result_str)
        return result

    return wrapper

path = '/Users/kfarid/Desktop/Education/MSc_Freiburg/work/RNA-benchmark/ViennaRNA-2.5.0/ViennaRNA250/bin/RNAfold'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def encode_struc(dots):
    s = []
    for i in dots:
        if i == '.':
            s.append(1)
        elif i == '(':
            s.append(2)
        elif i == ')':
            s.append(3)
    return s


def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def convert_to_list(base_seq):
    str_struc = []
    for i in base_seq:
        if i == 'A':
            str_struc.append(1)
        elif i == 'U':
            str_struc.append(2)
        elif i == 'G':
            str_struc.append(3)
        elif i == 'C':
            str_struc.append(4)
    #struc = ''.join(str_struc)
    return str_struc


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


LOCATION_FEATURES = 8
BASE_FEATURES = 9



class Eternabrain():
    def __init__(self,
                 structure: Optional[str] = None,
                 model_name = 'CNN18',
                 models_path = 'RnaBench/core/design_algorithms/EternaBrain/rna_prediction/models',
                 renderer: str = 'rnaplot',
                 max_iterations_factor: int = 10e6,
                 min_threshold: float = 0.6,
                 max_len: int = 400,):

        self.model_name = model_name
        self.model_path = models_path
        self.renderer = renderer

        self.max_iterations_factor = max_iterations_factor
        self.min_threshold = min_threshold
        self.max_len = max_len

        self.tf_shape = LOCATION_FEATURES * self.max_len
        self.base_shape = BASE_FEATURES * self.max_len
        self.len_longest = self.max_len

        if structure:
            self._target = structure


    def __name__(self):
        return 'EternaBrain'

    def infer(self, structure, ce=0.0, te=0.0):
        self._target = structure
        len_puzzle = len(self._target)
        NUCLEOTIDES = 'A' * len_puzzle
        ce = 0.0
        te = 0.0



        base_seq = (convert_to_list(NUCLEOTIDES)) + ([0] * (self.len_longest - len_puzzle))
        # cdb = '.'*len_puzzle
        current_struc = (encode_struc(RNA.fold(NUCLEOTIDES)[0])) + ([0] * (self.len_longest - len_puzzle))
        target_struc = encode_struc(self._target) + ([0] * (self.len_longest - len_puzzle))
        current_energy = [ce] + ([0] * (self.len_longest - 1))
        target_energy = [te] + ([0] * (self.len_longest - 1))
        current_pm = format_pairmap(NUCLEOTIDES) + ([0] * (self.len_longest - len_puzzle))
        target_pm = format_pairmap(self._target) + ([0] * (self.len_longest - len_puzzle))
        # locks = ([2]*32 + [1] * 85 + [2]*85) + ([0]*(len_longest - len_puzzle))
        locks = ([1] * len_puzzle) + ([0] * (self.len_longest - len_puzzle))

        # print len(base_seq),len(current_struc),len(DOT_BRACKET),len(target_struc),len(current_energy),len(target_energy),len(locks)

        inputs2 = np.array(
            [base_seq, current_struc, target_struc, current_energy, target_energy, current_pm, target_pm, locks])

        '''
        Change inputs when altering number of features
        '''
        # inputs2 = np.array([base_seq,current_energy,target_energy,current_pm,target_pm,locks])

        inputs = inputs2.reshape([-1, self.tf_shape])

        with tf.Graph().as_default() as base_graph:
            saver1 = tf.train.import_meta_graph(self.model_path + '/base/base' + self.model_name + '.meta')  # CNN15
        sess1 = tf.Session(
            graph=base_graph)  # config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        saver1.restore(sess1, self.model_path + f'/base/base' + self.model_name)

        x = base_graph.get_tensor_by_name('x_placeholder:0')
        y = base_graph.get_tensor_by_name('y_placeholder:0')
        keep_prob = base_graph.get_tensor_by_name('keep_prob_placeholder:0')

        base_weights = base_graph.get_tensor_by_name('op7:0')

        base_feed_dict = {x: inputs, keep_prob: 1.0}

        with tf.Graph().as_default() as location_graph:
            saver2 = tf.train.import_meta_graph(self.model_path + '/location/location' + self.model_name + '.meta')
        sess2 = tf.Session(
            graph=location_graph)  # config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        saver2.restore(sess2, self.model_path + '/location/location' + self.model_name)

        x2 = location_graph.get_tensor_by_name('x_placeholder:0')
        y2 = location_graph.get_tensor_by_name('y_placeholder:0')
        keep_prob2 = location_graph.get_tensor_by_name('keep_prob_placeholder:0')

        location_weights = location_graph.get_tensor_by_name('op7:0')

        print('models loaded')

        location_feed_dict = {x2: inputs, keep_prob2: 1.0}
        movesets = []
        iteration = 0
        reg = []
        for i in range(int(self.max_iterations_factor * len_puzzle)):
            if np.all(inputs2[1] == inputs2[2]):
                print("Puzzle Solved")
                break
            else:
                location_array = ((sess2.run(location_weights, location_feed_dict))[0])

                inputs2 = inputs.reshape([LOCATION_FEATURES, self.tf_shape // LOCATION_FEATURES])
                location_array = location_array[:len_puzzle] - min(location_array[:len_puzzle])
                total_l = sum(location_array)
                location_array = location_array / total_l
                # location_array = softmax(location_array)
                location_change = (choice(list(range(0, len(location_array))), 1, p=location_array, replace=False))[0]
                # location_change = np.argmax(location_array)
                la = [0.0] * self.len_longest
                la[location_change] = 1.0
                inputs2 = np.append(inputs2, la)
                inputs = inputs2.reshape([-1, self.base_shape])
                base_feed_dict = {x: inputs, keep_prob: 1.0}

                base_array = ((sess1.run(base_weights, base_feed_dict))[0])
                base_array = base_array - min(base_array)
                total = sum(base_array)
                base_array = base_array / total
                # base_array = softmax(base_array)

                # if np.random.rand() > 0.0:
                # FOR CHOOSING STOCHASTICALLY
                base_change = (choice([1, 2, 3, 4], 1, p=base_array, replace=False))[0]
                # else:
                # NOT STOCHASTICALLY
                # base_change = np.argmax(base_array) + 1

                inputs2 = inputs.reshape([BASE_FEATURES, self.base_shape // BASE_FEATURES])

                # if inputs2[0][location_change] == base_change:
                #     second = second_largest(base_array)
                #     base_change = np.where(base_array==second)[0][0] + 1

                temp = copy.deepcopy(inputs2[0])
                temp[location_change] = base_change
                move = [base_change, location_change]
                movesets.append(move)
                # print move
                str_seq = []
                for i in temp:
                    if i == 1:
                        str_seq.append('A')
                    elif i == 2:
                        str_seq.append('U')
                    elif i == 3:
                        str_seq.append('G')
                    elif i == 4:
                        str_seq.append('C')
                    else:
                        continue
                str_seq = ''.join(str_seq)
                str_struc, current_e = RNA.fold(str_seq)
                current_pm = format_pairmap(str_struc)
                print(str_struc)
                # print len(str_struc)
                print(similar(str_struc, self._target))
                rna_struc = []
                for i in inputs2[2]:
                    if i == 1:
                        rna_struc.append('.')
                    elif i == 2:
                        rna_struc.append('(')
                    elif i == 3:
                        rna_struc.append(')')
                    else:
                        continue
                rna_struc = ''.join(rna_struc)
                target_e = RNA.energy_of_structure(str_seq, rna_struc, 0)
                enc_struc = []
                for i in str_struc:
                    if i == '.':
                        enc_struc.append(1)
                    elif i == '(':
                        enc_struc.append(2)
                    elif i == ')':
                        enc_struc.append(3)
                    else:
                        continue
                inputs2[0] = temp
                inputs2[1][:len(enc_struc)] = (enc_struc)
                inputs2[3][0] = current_e
                inputs2[4][0] = target_e
                inputs2[5][:len(enc_struc)] = current_pm
                inputs_loc = inputs2[0:8]
                inputs = inputs_loc.reshape([-1, self.tf_shape])
                base_feed_dict = {x: inputs, keep_prob: 1.0}
                location_feed_dict = {x2: inputs, keep_prob2: 1.0}
                iteration += 1
                reg = []
                for i in inputs2[0]:
                    if i == 1:
                        reg.append('A')
                    elif i == 2:
                        reg.append('U')
                    elif i == 3:
                        reg.append('G')
                    elif i == 4:
                        reg.append('C')
                    else:
                        continue
                reg = ''.join(reg)
                # print inputs2[0][:len_puzzle]
                print(reg)
                print(iteration)
                # print current_struc[:len(enc_struc)]
                # print target_struc[:len(enc_struc)]
                # print inputs2[1][:len(enc_struc)]
                # print format_pairmap(str_struc)
                if similar(str_struc, self._target) >= self.min_threshold:
                    print('similar')
                    print(str_struc)
                    print(self._target)
                    print(reg)
                    break

        level1, m2, solved_sbc = sbc(self._target, reg)
        level2, m3, solved_dsp = dsp(self._target, level1, vienna_version=2)

        return level2, solved_dsp

    def design(self,
               structure: Optional[str] = None,

               ):

        if structure is not None:
            self._target = structure
        design, solved = self.infer(structure, '-1', 0.1)
        print(RNA.fold(design))
        return design, solved

    @save_timing
    def design_timed(self, structure: Optional[str] = None, *args, **kwargs
                                ):
        return self.design(structure=structure)


    def __call__(self, structure):
        return self.design(structure=structure)



if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument('--structure', default='.......((((...))))..................(((((......(((................))).....)))))........................', type=str)
    argparse.add_argument('--index', type=int, default=0)
    argparse.add_argument('--out_dir', type=str, default='eternabrain_results')
    args = argparse.parse_args()
    design_tool = Eternabrain()
    print(design_tool.design_timed(structure=args.structure, index=args.index, out_dir = args.out_dir))
