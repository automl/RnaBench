import RnaBench

from RnaBench.lib.visualization import RnaVisualizer

rna_vis = RnaVisualizer()

rna_vis.compare_performance_plot(results_dir='results/RNA_folding/inter_family',
                                 # latest=False,
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
                                 metrics=['f1_score', 'mcc', 'weisfeiler_lehman', 'recall', 'precision', 'f1_shifted'],  # , 'solved',
                                 best_three_key=None,  # 'weisfeiler_lehman',
                                 n_best=3,
                                 title=None,
                                 # range=None, #not there yet
                                 # fill=None, #not there yet
                                 log=False,
                                 fraction=1.,
                                 radial_axis_range=[0.5, 1.0],  # [0, 1.0],
                                 log_radial_axis_range=[-1,0],
                                 legend_only=False,
                                 radial_dtick=0.1,
                                 )


rna_vis.compare_performance_plot(results_dir='results/RNA_folding/inter_family',
                                 # latest=False,
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
                                 metrics=['f1_score', 'mcc', 'weisfeiler_lehman', 'recall', 'precision', 'f1_shifted'],  # , 'solved',
                                 best_three_key=None,  # 'weisfeiler_lehman',
                                 n_best=3,
                                 title=None,
                                 # range=None, #not there yet
                                 # fill=None, #not there yet
                                 log=False,
                                 fraction=1.,
                                 radial_axis_range=[0.7, 1.0],  # [0, 1.0],
                                 log_radial_axis_range=[-1,0],
                                 legend_only=False,
                                 radial_dtick=0.1,
                                 )


rna_vis.analyze_runtime_latest_runs(
                                    results_dir='results/RNA_folding/inter_family',
                                    metrics=None,  # ['f1_score', 'mcc', 'weisfeiler_lehman', 'recall', 'precision', 'specificity', 'solved', 'f1_shifted'], 'time', 'valid_gc_content', 'valid_energy', 'valid_8_U_sequence', 'valid_spacer_structure', 'valid_aptamer_sequence', 'valid_shape', 'valid_8-U_structure', 'valid_aptamer_structure', 'valid_co_folding', 'valid_sequence_and_structure', 'unique_valid_candidates', 'unique_valid_structures', 'valid_candidates'],
                                    renderer="browser", # "svg",
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