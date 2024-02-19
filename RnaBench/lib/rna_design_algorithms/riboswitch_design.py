import numpy as np

class OriginalProcedure():
    _complement_map = {
      'A': 'U',
      'C': 'G',
      'G': 'C',
      'U': 'A',
    }

    def __init__(self,
                 seed=0,
                 n_candidates=1000,
                 spacer_min_length=6,
                 spacer_max_length=20,
                 complement_pos_min=21,
                 complement_pos_max=32,
                 spacer_lib_size=None,
                 aptamer_sequence='AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA',
                 ):
        self.seed = seed
        self.num_candidates = n_candidates
        self.rng = np.random.default_rng(seed=self.seed)
        self.spacer_min_length = spacer_min_length
        self.spacer_max_length = spacer_max_length
        self.complement_pos_min = complement_pos_min
        self.complement_pos_max = complement_pos_max
        if spacer_lib_size is None:
            self.spacer_lib_size = n_candidates
        else:
            self.spacer_lib_size = spacer_lib_size
        self.aptamer_sequence = aptamer_sequence
        self.eight_u = 'UUUUUUUU'

    def __repr__(self):
        return "OriginalDesignByWachsmuth2012"

    def __name__(self):
        return "OriginalDesignByWachsmuth2012"

    def generate_spacer(self, nucs=['A', 'C', 'G', 'U']):
        spacer_size = self.rng.integers(low=self.spacer_min_length, high=self.spacer_max_length+1)
        spacer = self.rng.choice(nucs, size=spacer_size)
        return ''.join(spacer)

    def get_complement(self, seq):
        return ''.join([self._complement_map[s] for s in seq])[::-1]


    def get_spacer_lib(self):
        spacer_lib = []
        for i in range(self.spacer_lib_size):
            spacer = self.generate_spacer()
            spacer_lib.append(spacer)
        return spacer_lib

    def get_complement_lib(self):
        complement_lib = []
        for i in range(self.complement_pos_min, self.complement_pos_max+1):
            complement = self.get_complement(self.aptamer_sequence[i:])
            complement_lib.append(complement)
        return complement_lib


    def __call__(self):
        spacer_lib = self.get_spacer_lib()
        complement_lib = self.get_complement_lib()
        spacer_lib = list(set(spacer_lib))
        candidates = []
        for i in range(self.num_candidates):
            spacer = self.rng.choice(spacer_lib)
            complement = self.rng.choice(complement_lib)
            candidate = self.aptamer_sequence + spacer + complement + self.eight_u
            candidates.append(list(candidate))
        return candidates
