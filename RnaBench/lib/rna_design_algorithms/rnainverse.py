import subprocess
import numpy as np
from typing import Optional

class RNAInverse():
    def __init__(self,
                 structure: Optional[str] = None):
        if structure:
            self._target = structure

    def __name__(self):
        return 'RNAInverse'

    def __repr__(self):
        return 'RNAInverse'

    def design(self,
               structure: str,
               n_designs: int = 1,
               nc: bool = False,
               ):

        self._target = structure

        call = ["RNAinverse", f"-R{n_designs}"]

        if nc:
            call.append("--nsp")
            call.append("-GA,-AC,-UC,CC,AA,GG,UU,-NU,-NA,-NC,-NG")

        ps = subprocess.run(['echo', self._target], check=True, capture_output=True)
        design = subprocess.run(call, input=ps.stdout, capture_output=True)
        design = design.stdout.decode('utf-8').strip().split()
        if len(design) == 0:
            return ['A'] * len(structure)
        return design[0]

    def __call__(self, structure):
        return self.design(structure=structure)


# define main to get example working
if __name__ == '__main__':
    rna_inverse = RNAInverse()
    # get example structure
    structure = "((((((((((((..)))))))))))))"
    # design
    design = rna_inverse.design(structure=structure)
    print(design)
