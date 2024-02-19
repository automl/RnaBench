import subprocess

from typing import Optional
from pathlib import Path

import numpy as np

from RnaBench.lib.execution import timing
from RnaBench.lib.utils import db2pairs

rng = np.random.default_rng(seed=0)

NUCS = {
    'T': 'U',
    'P': 'U',
    'R': 'A',  # or 'G'
    'Y': 'C',  # or 'T'
    'M': 'C',  # or 'A'
    'K': 'U',  # or 'G'
    'S': 'C',  # or 'G'
    'W': 'U',  # or 'A'
    'H': 'C',  # or 'A' or 'U'
    'B': 'U',  # or 'G' or 'C'
    'V': 'C',  # or 'G' or 'A'
    'D': 'A',  # or 'G' or 'U'
    'N': rng.choice(['A', 'C', 'G', 'U']),  # 'N',
    'A': 'A',
    'U': 'U',
    'C': 'C',
    'G': 'G',
}

class PKiss():
    """
    """
    def __name__(self):
        return 'PKiss'

    def __repr__(self):
        return 'PKiss'

    def fold(self, sequence):
        if isinstance(sequence, float):
            return np.nan

        s = ''.join([NUCS[x] for x in sequence])  # can't handle IUPAC nucleotides
        output = subprocess.check_output(["pKiss", "--mode", "mfe", "--strategy", "A", s])
        output = output.decode('utf-8')
        structure = output.split()[-1]
        pairs = db2pairs(structure)
        return pairs

    @timing
    def fold_with_timing(self, sequence: str):
        s = ''.join([NUCS[x] for x in sequence])  # can't handle IUPAC nucleotides
        output = subprocess.check_output(["pKiss", "--mode", "mfe", "--strategy", "A", s])
        output = output.decode('utf-8')
        structure = output.split()[-1]
        return structure

    def __call__(self, sequence):
        return self.fold(sequence)


if __name__ == '__main__':
    sequence = 'AUGUAGUAGUAACAGCCGCGCUAGCAUCGUA'
    pkiss = PKiss()
    print(pkiss(sequence))
    print(pkiss.fold_with_timing(sequence))