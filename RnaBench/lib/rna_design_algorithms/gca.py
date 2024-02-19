from typing import Optional

class DeterministicGCA():
    def __init__(self,
                 structure: Optional[str] = None):
        if structure:
            self._target = structure

    def __name__(self):
        return 'DeterministicGCA'

    def __repr__(self):
        return 'DeterministicGCA'

    def design(self,
               structure: str,
               sequence: Optional[str] = None,
               ):

        self._target = structure

        design = ['A'] * len(self._target)

        for i, s in enumerate(self._target):
            if s in ['(', '[', '{', '<'] or s.isupper():
                design[i] = 'G'
            elif s in [')', ']', '}', '>'] or s.islower():
                design[i] = 'C'

        if sequence is not None:
            for i, s in enumerate(sequence):
                if s == 'N':
                    continue
                else:
                    design[i] = s

        return ''.join(design)


    def __call__(self, structure, sequence=None):
        return self.design(structure=structure, sequence=sequence)
