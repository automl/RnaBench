import subprocess

from typing import Optional, Union
from pathlib import Path

from RnaBench.lib.execution import timing
from RnaBench.lib.utils import db2pairs

class LinearFoldV():
    def __init__(self, linearfold_dir: str = 'external_algorithms/LinearFold'):
        self.cwd = Path(linearfold_dir)

    def __name__(self):
        return 'LinearFold-V'

    def __repr__(self):
        return 'LinearFold-V'


    def fold(self, sequence: str):
        if isinstance(sequence, float):
            return np.nan

        sequence = ''.join(sequence)
        ps = subprocess.Popen(['echo', sequence], stdout=subprocess.PIPE)
        structure = subprocess.check_output(['./linearfold', '-V'], stdin=ps.stdout, cwd=self.cwd)
        structure = structure.decode('utf-8').split()[1]
        pairs = db2pairs(structure)
        return pairs

    @timing
    def fold_with_timing(self, sequence: str):
        ps = subprocess.Popen(['echo', sequence], stdout=subprocess.PIPE)
        structure = subprocess.check_output(['./linearfold', '-V'], stdin=ps.stdout, cwd=self.cwd)
        structure = structure.decode('utf-8').split()[1]
        return structure

    def __call__(self, sequence):
        return self.fold(sequence)

class LinearFoldC():
    def __init__(self, linearfold_dir: str = 'external_algorithms/LinearFold'):
        self.cwd = Path(linearfold_dir)

    def __name__(self):
        return 'LinearFold-C'

    def __repr__(self):
        return 'LinearFold-C'


    def fold(self, sequence: str):
        if isinstance(sequence, float):
            return np.nan

        sequence = ''.join(sequence)
        ps = subprocess.Popen(['echo', sequence], stdout=subprocess.PIPE)
        structure = subprocess.check_output(['./linearfold'], stdin=ps.stdout, cwd=self.cwd)
        structure = structure.decode('utf-8').split()[1]
        pairs = db2pairs(structure)
        return pairs

    @timing
    def fold_with_timing(self, sequence: str):
        ps = subprocess.Popen(['echo', sequence], stdout=subprocess.PIPE)
        structure = subprocess.check_output(['./linearfold'], stdin=ps.stdout, cwd=self.cwd)
        structure = structure.decode('utf-8').split()[1]
        return structure


    def __call__(self, sequence):
        return self.fold(sequence)


if __name__ == '__main__':
    sequence = 'AUGUAGUAGUAACAGCCGCGCUAGCAUCGUA'
    linearfoldc = LinearFoldC()
    print(linearfoldc(sequence))
    print(linearfoldc.fold_with_timing(sequence))

    linearfoldv = LinearFoldV()
    print(linearfoldv(sequence))
    print(linearfoldv.fold_with_timing(sequence))