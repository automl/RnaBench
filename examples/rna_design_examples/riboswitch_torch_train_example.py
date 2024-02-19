"""
This example shows how to get and use a torch data iterator from RnaBench for the
RiboswitchDesign benchmark.
"""

import RnaBench
import torch
import time
import numpy as np

from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import db2pairs

device = 0 if torch.cuda.is_available() else 'cpu'
batch_size = 256

# instantiate the folding benchmark
folding_benchmark = RnaBench.RiboswitchDesignBenchmark()

# Get torch DataLoader for train set
# There is no valid or test set for the riboswitch design task
train_iterator, _, _ = folding_benchmark.get_iterators(
                                                 matrix=True,
                                                 device=device,
                                                 batch_size=batch_size,
                                                 )

print('### Start iterating', len(train_iterator.dataset), 'samples')

ts = time.time()

for i_batch, sampled_batch in enumerate(train_iterator):
    for b, length in enumerate(sampled_batch["length"].detach().cpu().numpy()):
        sequence_list = [train_iterator.dataset.seq_itos[i]
    for i in sampled_batch['sequence'][b, 1:length+1].detach().cpu().numpy()]  # +1 due to bos and eos...
        matrix = sampled_batch['matrix'][b, :length, :length].detach().cpu().numpy()
        pairs = [i.tolist()
    for i in sampled_batch['pairs'][b, :sampled_batch['num_pairs'][b]].detach().cpu().numpy()]
        for p1, p2, _ in pairs:
            assert matrix[p1, p2] == 1

te = time.time()

print('### Script runs', te - ts, 'seconds')



