import RnaBench
import pandas as pd
from pathlib import Path

from RnaBench.lib.visualization import RnaVisualizer
from RnaBench.lib.rna_folding_algorithms.rnafold import RNAFold
from RnaBench.lib.utils import pairs2db

vis = RnaVisualizer()

model = RNAFold()

benchmark = RnaBench.RnaFoldingBenchmark(task='inter_family',
                                         nc=False,
                                         pks=False,
                                         multiplets=False,
                                         )

def prediction_wrapper(task):
    prediction, _ = model(task.sequence)
    return prediction

metrics = benchmark(prediction_wrapper, algorithm_name='Representation_example')

plk_list = list(Path('results', 'RNA_folding', 'inter_family', 'Representation_example').glob('*.plk'))

df = pd.read_pickle(plk_list[0])

# print(df)

for i, row in df.iterrows():
    pred_pairs = row['predicted_pairs']
    true_pairs = row['pairs']
    length = row['length']
    sequence = row['sequence']
    true_structure = pairs2db(true_pairs, sequence)
    pred_structure = pairs2db(pred_pairs, sequence)


    vis.plot_matrices(pred_pairs=pred_pairs,
                      true_pairs=true_pairs,
                      length=length,
                      show=False,
                      id=i,
                      )

    vis.visualize_rna(true_pairs, sequence, f'true_structure_{i}', algorithm='RNAFold', plot_dir='plots/varna', plot_type='radiate', resolution='8.0')
    vis.visualize_rna(pred_pairs, sequence, f'predicted_structure_{i}', algorithm='RNAFold', plot_dir='plots/varna', plot_type='radiate', resolution='8.0')

    vis.visualize_rna(true_pairs, true_structure, f'true_structure_db_{i}', algorithm='RNAFold', plot_dir='plots/varna', plot_type='radiate', resolution='8.0')
    vis.visualize_rna(pred_pairs, pred_structure, f'predicted_structure_db_{i}', algorithm='RNAFold', plot_dir='plots/varna', plot_type='radiate', resolution='8.0')
