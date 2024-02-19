import RnaBench
import torch
import time
import numpy as np
import pandas as pd

from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import db2pairs

def prediction_wrapper(rna_folding_task, *args, **kwargs):
    pred_pairs, energy = model(rna_folding_task.sequence)
    return pred_pairs

print('### Start')
ts = time.time()
device = 0 if torch.cuda.is_available() else 'cpu'
batch_size = 8

print('### Instantiate Benchmark')
# instantiate the structure prediction benchmark
benchmark = RnaBench.RnaFoldingBenchmark(task='inter_family',
                                         max_length=200,
                                         )
# instantiate your model, here exemplarily using RNAfold
model = RNAFold()

print('### Get iterators')
# Get torch DataLoader for train, valid, test
train_iterator, valid_iterator, test_iterator = benchmark.get_iterators(
                                                     matrix=True,
                                                     device=device,
                                                     batch_size=batch_size,
                                                     pks=True,
                                                     multiplets=True,
                                                     nc=True,
                                                     max_length=200,
                                                     )

print('### Start training on', len(train_iterator.dataset), 'samples')

train_ids = []

for i_batch, sampled_batch in enumerate(train_iterator):
    for b, length in enumerate(sampled_batch["length"].detach().cpu().numpy()):
        sequence_list = [train_iterator.dataset.seq_itos[i]
    for i in sampled_batch['sequence'][b, 1:length+1].detach().cpu().numpy()]  # +1 due to bos and eos...
        idx = sampled_batch['task_id'][b].detach().cpu().numpy()

        # Do some training in here

        train_ids.append(idx)

print('### Processed', len(train_ids), 'training samples')

validation_results = benchmark(prediction_wrapper,
                               dataset=valid_iterator.dataset,
                               save_results=False,
                               algorithm_name='RNAFold',
                               )

print('### Validation_results for', len(valid_iterator.dataset), 'samples')
print(validation_results)

ft_train_iterator, ft_valid_iterator, ft_test_iterator = benchmark.get_iterators(
                                                matrix=True,
                                                device=device,
                                                batch_size=batch_size,
                                                task='inter_family_fine_tuning',
                                                max_length=200,
                                                )

fine_tuning_training_scores = []
for i_batch, sampled_batch in enumerate(ft_train_iterator):
    for b, length in enumerate(sampled_batch["length"].detach().cpu().numpy()):
        sequence_list = [train_iterator.dataset.seq_itos[i]
    for i in sampled_batch['sequence'][b, 1:length+1].detach().cpu().numpy()]  # +1 due to bos and eos...
        idx = sampled_batch['task_id'][b].detach().cpu().numpy()

        # Do some fine-tuning in here
        # here is an example using RNAFold (this is not really fine tuning :-)).

        # score = benchmark(prediction_wrapper,
        #                   dataset=ft_train_iterator.dataset,
        #                   task_ids=idx,
        #                   save_results=False,
        #                   algorithm_name='RNAFold',
        #                   )

        # fine_tuning_training_scores.append(score)


# print('### Fine tuning training scores')
# scores = pd.DataFrame.from_dict(fine_tuning_training_scores)
# print(scores.mean())

# validation_results = benchmark(prediction_wrapper,
#                                dataset=ft_valid_iterator.dataset,
#                                save_results=False)
#
# print('### Validation_results for', len(ft_valid_iterator.dataset), 'samples')
# print(validation_results)

benchmark_results = benchmark(prediction_wrapper)

print('### Benchmark results for', len(ft_test_iterator.dataset), 'samples')
print(benchmark_results)

te = time.time()
print('### Script runs', te - ts, 'seconds')