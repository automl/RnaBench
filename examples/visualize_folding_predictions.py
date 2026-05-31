import RnaBench

from pathlib import Path

from RnaBench.lib.visualization import RnaVisualizer

rna_vis = RnaVisualizer()


def find_results_dir():
    """Return the first non-empty results dir under results/RNA_folding/*."""
    for task in ('intra_family', 'inter_family', 'biophysical_model'):
        p = Path('results', 'RNA_folding', task)
        if p.exists() and any(p.iterdir()):
            return str(p)
    raise FileNotFoundError("No results/RNA_folding/<task> directory found. "
                            "Run an evaluation script first.")


results_dir = find_results_dir()
print(f"### Plotting results from {results_dir}")

rna_vis.compare_performance_plot(results_dir=results_dir,
                                 show=True,
                                 out_dir='plots',
                                 output_format='pdf',
                                 legend=True,
                                 nc=True,
                                 pks=True,
                                 multiplets=True,
                                 min_length=None,
                                 max_length=None,
                                 pk_only=False,
                                 multiplets_only=False,
                                 nc_only=False,
                                 metrics=['f1_score', 'mcc', 'weisfeiler_lehman', 'recall', 'precision', 'f1_shifted'],
                                 best_three_key=None,
                                 n_best=3,
                                 title=None,
                                 log=False,
                                 fraction=1.,
                                 radial_axis_range=[0.5, 1.0],
                                 log_radial_axis_range=[-1, 0],
                                 legend_only=False,
                                 radial_dtick=0.1,
                                 )


rna_vis.compare_performance_plot(results_dir=results_dir,
                                 show=True,
                                 out_dir='plots',
                                 output_format='pdf',
                                 legend=True,
                                 nc=False,
                                 pks=False,
                                 multiplets=False,
                                 min_length=None,
                                 max_length=None,
                                 pk_only=False,
                                 multiplets_only=False,
                                 nc_only=False,
                                 metrics=['f1_score', 'mcc', 'weisfeiler_lehman', 'recall', 'precision', 'f1_shifted'],
                                 best_three_key=None,
                                 n_best=3,
                                 title=None,
                                 log=False,
                                 fraction=1.,
                                 radial_axis_range=[0.7, 1.0],
                                 log_radial_axis_range=[-1, 0],
                                 legend_only=False,
                                 radial_dtick=0.1,
                                 )


rna_vis.analyze_runtime_latest_runs(
    results_dir=results_dir,
    metrics=None,
    renderer="browser",
    show=True,
    out_dir='plots',
    output_format='pdf',
    legend=False,
    nc=True,
    pks=True,
    multiplets=True,
    min_length=None,
    max_length=None,
    pk_only=False,
    multiplets_only=False,
    nc_only=False,
    best_three_key=None,
    n_best=3,
    file_list=None,
    title=None,
)
