import numpy as np
from pathlib import Path

from RnaBench.lib.visualization import RnaVisualizer
from RnaBench.lib.metrics import GoalDirectedMetrics
from RnaBench.lib.utils import db2pairs

original_structure = '(((((((((....((((((((.......((((............))))........)))))).)).(((((......((.((.((......)))).)).....))))).))))))))).................'
original_sequence = 'GCCAACGACCAUACCACGCUGAAUACAUCGGUUCUCGUCCGAUCACCGAAAUUAAGCAGCGUCGGGCGCGGUUAGUACUUAGAUGGGGGACCGCUUGGGAACACCGCGUGUUGUUGGCCUCGUCCACAACUUUUU'
original_pairs = db2pairs(original_structure)

shift_by_one = '.(((((((((....((((((((.......((((............))))........)))))).)).(((((......((.((.((......)))).)).....))))).)))))))))................'
shift_by_two = '..(((((((((....((((((((.......((((............))))........)))))).)).(((((......((.((.((......)))).)).....))))).)))))))))...............'

met = ['f1_score', 'wl', 'mcc', 'f1_shifted']

metrics = GoalDirectedMetrics(metrics=met)
plot_dir = 'plots/varna'
algorithm = 'none'
plot_type = 'radiate'

vis = RnaVisualizer()

structures = [original_structure, shift_by_one, shift_by_two]
labels = ['Ground Truth', 'Shift_one', 'Shift_two']
final_labels = []
paths = []
for structure, label in zip(structures, labels):
    pairs = db2pairs(structure)
    row = {
        'sequence': original_sequence,
        'predicted_sequence': original_sequence,
        'pairs': original_pairs,
        'predicted_pairs': pairs,
        'time': 0,
        'length': len(original_sequence)
    }
    results = metrics(row)
    f1 = results['f1_score']
    f1_shifted = results['f1_shifted']
    mcc = results['mcc']
    wl = results['weisfeiler_lehman']

    vis.visualize_rna(pairs=pairs,
                      sequence=original_sequence,
                      Id=label,
                      algorithm=algorithm,
                      plot_dir=plot_dir,
                      plot_type=plot_type,
                      resolution='20.0',
                      )

    l = f"F1: {np.round(f1, 3)}\n"+f"Shifted-F1: {np.round(f1_shifted, 3)}\n"+ f"MCC: {np.round(mcc, 3)}\n"+ f"WL: {np.round(wl, 3)}"
    final_labels.append(l)

    paths.append(Path(plot_dir, algorithm, f"{algorithm}_{label}_{plot_type}.png").resolve())

print(final_labels)

vis.visualize_rnas_from_paths(
    path_list=paths,
    titles=final_labels,
    rows=1,
    cols=3,
    rnas_per_plot=3,
    data_dir=plot_dir,
    fontsize=20,
)
