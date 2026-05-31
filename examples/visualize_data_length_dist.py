import pandas as pd

from RnaBench.lib.visualization import RnaVisualizer

# The histo_dataset_comparison_express plot only consumes the 'length' column,
# so we read it directly from the pickled DataFrame and skip the per-sample
# RNAStatistics aggregation (which iterates every pair and OOM's on
# biophysical_model_train's 600 k rows).

benchmarks = [
    'intra_family',
    'inter_family',
    'biophysical_model',
    'inverse_rna_folding',
    'constrained_design',
]


def length_df(path):
    return pd.read_pickle(path)[['length']].copy()


for benchmark in benchmarks:
    labels = []
    df_list = []

    df_list.append(length_df(f'data/{benchmark}_train.plk.gz'))
    labels.append(f"{benchmark}-Train")

    df_list.append(length_df(f'data/{benchmark}_valid.plk.gz'))
    labels.append(f"{benchmark}-Valid")

    df_list.append(length_df(f'data/{benchmark}_benchmark.plk.gz'))
    labels.append(f"{benchmark}-Test")

    if benchmark == 'inter_family':
        df_list.append(length_df(f'data/{benchmark}_fine_tuning_train.plk.gz'))
        labels.append(f"{benchmark}-Fine-Tune")

    vis = RnaVisualizer()
    vis.histo_dataset_comparison_express(df_list, labels=labels, show=False)
