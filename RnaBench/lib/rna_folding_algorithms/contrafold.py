import subprocess
import os
from pathlib import Path

from typing import Optional, Union

from RnaBench.lib.execution import timing
from RnaBench.lib.utils import db2pairs

class ContraFold():
    def __init__(self,
                 working_dir: str = 'working_dir'
                 ):
        self.working_dir = working_dir
        Path(working_dir).mkdir(exist_ok=True, parents=True)

    def __name__(self):
        return 'ContraFold'

    def __repr__(self):
        return 'ContraFold'

    def fold(self, sequence, id=None):
        if isinstance(sequence, float):
            return np.nan
        sequence = ''.join(sequence)
        if id is not None:
            infile = Path(self.working_dir, f'contrafoldin_{id}.rna')
        else:
            infile = Path(self.working_dir, 'contrafoldin.rna')
        infile.touch()
        with open(infile, 'w') as f:
            p = subprocess.run(["echo", '>id\n' + sequence], stdout=f)
        output = subprocess.check_output(["contrafold", "predict", str(infile.resolve())])  # , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        structure = output.decode('utf-8').split()[-1]
        pairs = db2pairs(structure)
        return pairs

    @timing
    def fold_with_timing(self, sequence, id=None):
        if id is not None:
            infile = Path(self.working_dir, f'contrafoldin_{id}.rna')
        else:
            infile = Path(self.working_dir, 'contrafoldin.rna')
        infile.touch()
        with open(infile, 'w') as f:
            p = subprocess.run(["echo", '>id\n' + sequence], stdout=f)
        output = subprocess.check_output(["contrafold", "predict", str(infile.resolve())])  # , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        structure = output.decode('utf-8').split()[-1]
        return structure


    def __call__(self, sequence: str, id=None):
        return self.fold(sequence, id)


if __name__ == '__main__':
    sequence = 'AUGUAGUAGUAACAGCCGCGCUAGCAUCGUA'
    contrafold = ContraFold()
    print(contrafold(sequence))
    print(contrafold.fold_with_timing(sequence))
