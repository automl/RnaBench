import RNA
import subprocess
import numpy as np
from typing import Optional
from RnaBench.lib.execution import timing
from RnaBench.lib.utils import db2pairs

class RNAFold():
    def __name__(self):
        return 'RNAFold'

    def __repr__(self):
        return 'RNAFold'

    def __call__(self, sequence):
        if isinstance(sequence, float):
            return np.nan, np.nan
        sequence = ''.join(sequence)
        db, energy = RNA.fold(sequence)
        pairs = db2pairs(db)
        return pairs, energy

        # return   # [1] should be energy

    @timing
    def fold_with_timing(self, sequence):
        return RNA.fold(sequence)[0]  # [1] should be energy


if __name__ == '__main__':
    rnafold = RNAFold()
    print(rnafold('ACGUCGCUAGCUAGCUAGCUA'))
    print(rnafold.fold_with_timing('ACGUCGCUAGCUAGCUAGCUA'))