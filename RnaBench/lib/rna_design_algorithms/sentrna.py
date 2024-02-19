
from typing import Optional
import numpy as np
import random
import os
import pickle
from SentRNA.SentRNA.util.compute_mfe import *
from SentRNA.SentRNA.util.draw_rna import *
from SentRNA.SentRNA.util.mutinf import *
from SentRNA.SentRNA.util.refine_moves import *
import argparse
from SentRNA.SentRNA.util.featurize_util import *
from SentRNA.SentRNA.util.feedforward import *

class SentRNA():
    def __init__(self,
                 structure: Optional[str] = None,
                 model_name: str = '/Users/kfarid/Desktop/Education/MSc_Freiburg/work/RNA-benchmark/RnaBench/core/design_algorithms/SentRNA/models/trained_models/matlot/test/MI-54_trial4',
                 renderer: str = 'rnaplot'):

        self.model_name = model_name
        self.renderer = renderer

        if structure:
            self._target = structure

    def __name__(self):
        return 'SentRNA'

    def infer(self, dataset, model, puzzle_name, MI_tolerance):
        self._target = dataset
        test_puzzles = [dataset]
#        t#est_puzzles = [i[0] for i in dataset]
        model_name = model[model.index('test') + 5:]
        base_dir = model[:model.index('test')][:-1]
        if base_dir == '':
            base_dir = '.'
        for i in os.listdir(model):
            if '.data' in i:
                model_path = i[:i.index('.data')]
                test_model_path = '%s/%s' % (model, model_path)
                f'{base_dir}/results/MI_features_list.{model_name}.pkl'.encode()

        MI_features_list = pickle.load(open(f'{base_dir}/results/MI_features_list.{model_name}.pkl'.encode(), 'rb'))
        # '%s/results/layer_sizes.%s.pkl' % (base_dir, model_name)
        layer_sizes = pickle.load(open(f'{base_dir}/results/layer_sizes.{model_name}.pkl'.encode(), 'rb'))
        model = TensorflowClassifierModel(layer_sizes=layer_sizes)
        output = []
        for puzzle in test_puzzles:
            if puzzle_name != '-1' and puzzle != puzzle_name:
                continue

            dot_bracket = puzzle
            seq = ''.join(['A']*len(dot_bracket))

            fixed_bases = []
            solution, _ = model.evaluate(dot_bracket, seq, fixed_bases, layer_sizes, MI_features_list, test_model_path,
                                         refine=False, MI_tolerance=MI_tolerance, \
                                         renderer=self.renderer)
            accuracy = check_answer(solution, dot_bracket)
            solution2, _ = model.evaluate(dot_bracket, solution, fixed_bases, layer_sizes, MI_features_list,
                                          test_model_path, refine=True, MI_tolerance=MI_tolerance, \
                                          renderer=self.renderer)
            accuracy2 = check_answer(solution2, dot_bracket)
            if accuracy2 > accuracy:
                accuracy = accuracy2
                solution = solution2
            output.append([puzzle, dot_bracket, solution, accuracy])
            print([puzzle, dot_bracket, solution, accuracy])
        #pickle.dump(output, open('test_results/%s' % (results_path), 'w'))
        return solution

    def design(self,
               structure: Optional[str] = None,

               ):

        if structure is not None:
            self._target = structure
        design = self.infer(structure, self.model_name, '-1', 0.1)
        #design = ['A'] * len(self._target)

        return design

    def __call__(self, structure):
        return self.design(structure=structure)



if __name__ == '__main__':
    design_tool = SentRNA()
    print(design_tool.design('(((((...)))))'))
