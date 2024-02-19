import subprocess

from typing import Optional
from pathlib import Path
from RnaBench.lib.execution import timing
from RnaBench.lib.utils import db2pairs

class IpKnot():
    def __init__(self,
                 ipknot_dir: str = 'external_algorithms',
                 working_dir: str = 'working_dir'):
        self.cwd = ipknot_dir
        self.working_dir = working_dir

    def __name__(self):
        return 'ipknot'

    def __repr__(self):
        return 'ipknot'


    def fold(self, sequence: str, id=None):
        if isinstance(sequence, float):
            return np.nan

        sequence = ''.join(sequence)
        if id is not None:
            infile = Path(self.working_dir, f'ipknot_infile_{id}.rna')
        else:
            infile = Path(self.working_dir, 'ipknot_infile.rna')
        infile.touch()
        with open(infile, 'w') as f:
            p = subprocess.run(["echo", '>id\n' + sequence], stdout=f)
        output = subprocess.check_output(["./ipknot", "-i", str(infile.resolve())], cwd=self.cwd)
        output = output.decode('utf-8')
        structure = output.split()[-1]
        pairs = db2pairs(structure)
        return pairs

    @timing
    def fold_with_timing(self, sequence: str, id=None):
        if id is not None:
            infile = Path(self.working_dir, f'ipknot_infile_{id}.rna')
        else:
            infile = Path(self.working_dir, 'ipknot_infile.rna')
        infile.touch()
        with open(infile, 'w') as f:
            p = subprocess.run(["echo", '>id\n' + sequence], stdout=f)
        output = subprocess.check_output(["./ipknot", "-i", str(infile.resolve())], cwd=self.cwd)
        output = output.decode('utf-8')
        structure = output.split()[-1]
        return structure

    def __call__(self, sequence, id=None):
        return self.fold(sequence)

if __name__ == '__main__':
    sequence = 'AUGUAGUAGUAACAGCCGCGCUAGCAUCGUA'
    ipknot = IpKnot()
    print(ipknot(sequence))
    print(ipknot.fold_with_timing(sequence))