import RnaBench

from RnaBench.lib.visualization import RnaVisualizer

rna_vis = RnaVisualizer()

rna_vis.compare_performance_plot(results_dir='results/RNA_design/inverse_rna_folding/DeterministicGCA',
                                 # latest=False,
                                 show=True,
                                 out_dir='plots',
                                 output_format='pdf',
                                 legend=True,
                                 nc=True,
                                 pks=False,
                                 multiplets=True,
                                 min_length=None,
                                 max_length=None,
                                 pk_only=False,
                                 multiplets_only=False,
                                 nc_only=False,
                                 metrics=['f1_score', 'mcc', 'weisfeiler_lehman', 'recall', 'precision', 'f1_shifted', 'solved'],  # , 'solved',
                                 best_three_key=None,  # 'weisfeiler_lehman',
                                 n_best=3,
                                 title=None,
                                 # range=None, #not there yet
                                 # fill=None, #not there yet
                                 log=False,
                                 fraction=1.,
                                 radial_axis_range=[0.0, 1.0],  # [0, 1.0],
                                 log_radial_axis_range=[-1,0],
                                 legend_only=False,
                                 radial_dtick=0.2,
                                 )
