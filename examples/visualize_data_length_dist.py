from RnaBench.lib.datasets import RnaDataset
from RnaBench.lib.visualization import RnaVisualizer, RNAStatistics
# benchmark = 'intra_family'
# benchmark = 'inter_family'
# benchmark = 'biophysical_model'
# benchmark = 'inverse_rna_folding'
benchmark = 'constrained_design'

benchmarks = [
'intra_family',
'inter_family',
'biophysical_model',
'inverse_rna_folding',
'constrained_design',
]

for benchmark in benchmarks:

    labels = []
    df_list = []

    df_path=f'data/{benchmark}_train.plk.gz'
    dset = RnaDataset(dataset=df_path)
    stats = RNAStatistics(dset)
    stats.get_dataset_statistics()
    labels.append(f"{benchmark}-Train")
    df_list.append(stats.per_sample_stats_df)

    df_path=f'data/{benchmark}_valid.plk.gz'
    dset = RnaDataset(dataset=df_path)
    stats = RNAStatistics(dset)
    stats.get_dataset_statistics()
    labels.append(f"{benchmark}-Valid")
    df_list.append(stats.per_sample_stats_df)

    df_path=f'data/{benchmark}_benchmark.plk.gz'
    dset = RnaDataset(dataset=df_path)
    stats = RNAStatistics(dset)
    stats.get_dataset_statistics()
    labels.append(f"{benchmark}-Test")
    df_list.append(stats.per_sample_stats_df)

    if benchmark == 'inter_family':
        df_path=f'data/{benchmark}_fine_tuning_train.plk.gz'
        dset = RnaDataset(dataset=df_path)
        stats = RNAStatistics(dset)
        stats.get_dataset_statistics()
        labels.append(f"{benchmark}-Fine-Tune")
        df_list.append(stats.per_sample_stats_df)

    vis = RnaVisualizer()
    vis.histo_dataset_comparison_express(df_list,
                                         labels=labels,
                                         )
