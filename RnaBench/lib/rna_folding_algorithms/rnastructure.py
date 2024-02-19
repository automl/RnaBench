import subprocess
import os
import numpy as np
from pathlib import Path

from typing import Optional, Union

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


class Fold():
    def __init__(self,
                 working_dir: Union[Path, str] = 'working_dir',
                 sequence: Optional[str] = None):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True, parents=True)
        self.sequence = sequence
        # self.set_table_path()

    def __name__(self):
        return 'RNAStructure-Fold'

    def __repr__(self):
        return 'RNAStructure-Fold'

    def set_table_path(self):
        with open('rnastructure_datapath.txt') as f:
            datapath = f.readline().rstrip()
        os.environ['DATAPATH'] = datapath

    def fold(self, sequence, id=None):
        if isinstance(sequence, float):
            return np.nan, np.nan

        sequence = ''.join(sequence)
        if id is not None:
            infile = Path(self.working_dir, f'rnastructure_fold_infile_{id}.rna')
        else:
            infile = Path(self.working_dir, 'rnastructure_fold_infile.rna')
        infile.touch()
        ct_outfile = Path(self.working_dir, 'ct_outfile.ct')
        db_outfile = Path(self.working_dir, 'db_outfile.db')
        with open(infile, 'w') as f:
            p = subprocess.run(["echo", sequence], stdout=f)
        p = subprocess.run(["Fold", str(infile.resolve()), str(ct_outfile.resolve()), "-mfe"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        p = subprocess.run(["ct2dot", str(ct_outfile.resolve()), "1", str(db_outfile.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        structure = db_outfile.read_text().rstrip().split('\n')[-1]
        pairs = db2pairs(structure)
        return pairs

    @timing
    def fold_with_timing(self, sequence, id=None):
        if id is not None:
            infile = Path(self.working_dir, f'rnastructure_fold_infile_{id}.rna')
        else:
            infile = Path(self.working_dir, 'rnastructure_fold_infile.rna')
        infile.touch()
        ct_outfile = Path(self.working_dir, 'ct_outfile.ct')
        db_outfile = Path(self.working_dir, 'db_outfile.db')
        with open(infile, 'w') as f:
            p = subprocess.run(["echo", sequence], stdout=f)
        p = subprocess.run(["Fold", str(infile.resolve()), str(ct_outfile.resolve()), "-mfe"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        p = subprocess.run(["ct2dot", str(ct_outfile.resolve()), "1", str(db_outfile.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        structure = db_outfile.read_text().rstrip().split('\n')[-1]
        return structure


    def __call__(self, sequence: str, id=None):
        return self.fold(sequence)


if __name__ == '__main__':
    sequence = 'AUGUAGUAGUAACAGCCGCGCUAGCAUCGUA'
    rnastructure = Fold()
    print(rnastructure(sequence))
    print(rnastructure.fold_with_timing(sequence))