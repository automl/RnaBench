import time
import os
import subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

import matplotlib.image as mpimg
import matplotlib.colors as mcolors

from pathlib import Path
from itertools import product, combinations
from collections import Counter, defaultdict
from typing import Dict
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_circles

from mpl_toolkits.axes_grid1 import make_axes_locatable
from upsetplot import plot

from RnaBench.lib.utils import (
get_multiplets,
type_pairs,
lone_pair,
ct_file_output,
get_pair_types,
pairs2mat,
)

WIDTH_PT = 397.48499 # NeurIPS width
LEGEND_FONTSIZE = 30  # paper: 70
TICK_FONTSIZE = 30  # paper: 40 #
LABEL_FONTSIZE = 30 # paper: 68
TITEL_FONTSIZE = 15
DPI = 300
GRID_WIDTH = 2
LINE_WIDTH = 3
LEGEND_ITEM_SIZE = 30  # paper: 70


histo_colors = {'hamming': plt.cm.ocean,
                'metrics': plt.cm.RdYlGn,
                'solved': plt.cm.RdYlGn,
                'solved_struct': plt.cm.RdYlGn,
                'topology_solved': plt.cm.RdYlGn,
                'same_pks': plt.cm.RdYlGn,
                'same_topology': plt.cm.RdYlGn,
                'op4cl': plt.cm.ocean,
                'op4dot': plt.cm.ocean,
                'cl4op': plt.cm.ocean,
                'cl4dot': plt.cm.ocean,
                'dot4op': plt.cm.ocean,
                'dot4cl': plt.cm.ocean,
                'A_fault': plt.cm.ocean,
                'C_fault': plt.cm.ocean,
                'G_fault': plt.cm.ocean,
                'U_fault': plt.cm.ocean,
                'all': plt.cm.RdYlGn,
                'f1_score': plt.cm.RdYlGn,
                'mcc': plt.cm.RdYlGn,
                'precision': plt.cm.RdYlGn,
                'recall': plt.cm.RdYlGn,
                'norm_hamming': plt.cm.ocean}

plotly_colors = {
  'Pastel1': px.colors.qualitative.Pastel1,
  'G10': px.colors.qualitative.G10,
  'T10': px.colors.qualitative.T10,
  'Pastel': px.colors.qualitative.Pastel,
}

canonical_pairs = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']


def set_size_px(width_pt=WIDTH_PT, fraction=1, subplots=(1, 1)):
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    pixels_per_pt = DPI / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = ( (5 ** 0.5 - 1) / 2)

    # Figure width in inches
    fig_width_px = fig_width_pt * pixels_per_pt
    # Figure height in inches
    fig_height_px = fig_width_px * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_px, fig_height_px


class RnaVisualizer():
    def __init__(self,
                 # data: Dict[str, list],
                 ):
        pass
        # self.data = data

    def visualize_dataset(self):
        pass

    def visualize_performance(self):
        pass

    def visualize_rna(self, pairs, sequence, Id, algorithm='folding_algorithm', plot_dir='plots', plot_type='radiate', resolution='8.0'):
        Path(plot_dir, algorithm).mkdir(exist_ok=True, parents=True)
        multiplets = get_multiplets(pairs)
        watson_pairs, wobble_pairs, noncanonical_pairs = type_pairs(pairs, sequence)
        lone_bp = lone_pair(pairs)
        tertiary_bp = multiplets + noncanonical_pairs + lone_bp
        tertiary_bp = [list(x) for x in set(tuple(x) for x in tertiary_bp)]

        str_tertiary = []
        for i,I in enumerate(tertiary_bp):
            if i==0:
                str_tertiary += ('(' + str(I[0]+1) + ',' + str(I[1]+1) + '):color=""#FFFF00""')
            else:
                str_tertiary += (';(' + str(I[0]+1) + ',' + str(I[1]+1) + '):color=""#FFFF00""')

        tertiary_bp = ''.join(str_tertiary)

        ct_path = Path(plot_dir, algorithm, f"{algorithm}_{Id}.ct")

        ct_file_output(pairs, sequence, Id, ct_path, algorithm=algorithm)

        varna_path = str(Path(plot_dir, algorithm, f"{algorithm}_{Id}").resolve())
        if plot_type == 'radiate':
            subprocess.Popen(["varna", '-i', str(ct_path.resolve()), '-o', varna_path + '_radiate.png', '-algorithm', 'radiate', '-resolution', resolution, '-bpStyle', 'lw', '-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        elif plot_type == 'line':
            subprocess.Popen(["varna", '-i', str(ct_path.resolve()), '-o', varna_path + '_line.png', '-algorithm', 'line', '-resolution', resolution, '-bpStyle', 'lw', '-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    def visualize_rnas_from_paths(self, path_list, data_dir, titles=None, fontsize=20, rnas_per_plot=2, rows=1, cols=2, plot_dir='plots', plot_type='radiate', pgfplots=False, show=True):
#         if plot_type == 'radiate':
#             path_list = list(Path(data_dir).glob('*_radiate.png'))
#         elif plot_type == 'line':
#             path_list = list(Path(data_dir).glob('*_line.png'))
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")

        outpath = Path(data_dir, f"multiplot_{'-'.join(now.split('/')).replace(' ', '-')}.png")
        # print(rows, cols)
        if pgfplots:
            width = 5.5  # FOr neurips setup ?
            height = (width * cols) / rows
        else:
            # if knowing height compute width with
            height = 20
            width = height * (rows/cols)
        multiplicator = 0
        cur_row = 0
        sub = 0
        fig, axs = plt.subplots(rows, cols)  # figsize=(20, 4)  #constrained_layout=True, , tight_layout=True -> incompatible with constrained layout)  # 16)

        for i, p in enumerate(path_list):
            # print('### Plotting ', p)
            varna = mpimg.imread(p)
            cur_col = i - multiplicator * sub
            # print(cur_col)
            if cur_col >= cols:
                sub = cols
                multiplicator += 1
                cur_row += 1
                cur_col = i - multiplicator * sub
            # print(cur_row)
            if rows > 1:
                axs[cur_row][cur_col].imshow(varna, aspect=1)
                if cur_row == 0:
                    axs[cur_row][cur_col].sharey(axs[cur_row][0])
                    axs[cur_row][cur_col].autoscale()
                axs[cur_row][cur_col].spines['top'].set_visible(False)
                axs[cur_row][cur_col].spines['right'].set_visible(False)
                axs[cur_row][cur_col].spines['left'].set_visible(False)
                axs[cur_row][cur_col].spines['bottom'].set_visible(False)
                axs[cur_row][cur_col].axis('off')
                if titles is not None:
                    axs[cur_row][cur_col].set_title(titles[i], fontsize=fontsize, loc='center', verticalalignment='top',fontweight='bold')


            else:
                if cur_col != 0:
                    axs[cur_col].sharey(axs[cur_col-1])
                    axs[cur_col].autoscale()

                axs[cur_col].imshow(varna)
                axs[cur_col].spines['top'].set_visible(False)
                axs[cur_col].spines['right'].set_visible(False)
                axs[cur_col].spines['left'].set_visible(False)
                axs[cur_col].spines['bottom'].set_visible(False)
                axs[cur_col].axis('off')
                if titles is not None:
                    axs[cur_col].set_title(titles[i], fontsize=fontsize, loc='center', horizontalalignment='left', verticalalignment='top')  # ,fontweight='bold')

            # if unbalanced:
            #     axs[rows-1][cols-1].spines['top'].set_visible(False)
            #     axs[rows-1][cols-1].spines['right'].set_visible(False)
            #     axs[rows-1][cols-1].spines['left'].set_visible(False)
            #     axs[rows-1][cols-1].spines['bottom'].set_visible(False)
            #     axs[rows-1][cols-1].axis('off')

        fig.set_size_inches(height,width,forward=False)
        plt.subplots_adjust(hspace=-0.2, wspace=0.001)  # previous hspace = 0.001 or 0.0001
        # plt.rc(usetex=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        print('### Save figure to', outpath.resolve())
        if show:
            plt.show()
        fig.savefig(outpath, format='png', dpi=300, bbox_inches='tight')  #  dpi=300, bbox_inches='tight') #  -> incompatible with constrained layout
        plt.close('all')



    def visualize_multiple_rnas(self, data_dir, rnas_per_plot=10, plot_dir='plots', plot_type='radiate'):
        if plot_type == 'radiate':
            path_list = list(Path(data_dir).glob('*_radiate.png'))
        elif plot_type == 'line':
            path_list = list(Path(data_dir).glob('*_line.png'))

        rows = rnas_per_plot // 5
        cols = rnas_per_plot // 2
        print(rows, cols)
        split_number = len(path_list) // rnas_per_plot
        path_list = np.array_split(path_list, split_number)
        # print(path_list.shape)
        # path_list = [np.array_split(l, rows) for l in path_list]
        # print(path_list[0])
        for varna_paths_list in path_list:
            # print()
            # print(varna_paths_list)
            outpath = Path(data_dir, '_'.join([str(x.stem).split('_')[-2] for x in varna_paths_list]) + '_multiplot.png')
            varna_paths_list = np.array_split(varna_paths_list, rows)
            # print(varna_paths_list)
            if pgfplots:
                width = 5.5  # FOr neurips setup ?
                height = (width * cols) / rows
            else:
                # if knowing height compute width with
                height = 20
                width = height * (rows/cols)
            multiplicator = 0
            cur_row = 0
            sub = 0
            fig, axs = plt.subplots(rows, cols)  # figsize=(20, 4)  #constrained_layout=True, , tight_layout=True -> incompatible with constrained layout)  # 16)

            for cur_row, varna_paths in enumerate(varna_paths_list):
                # print()
                # print(varna_paths)
                for i, p in enumerate(varna_paths):
                    print('### Plotting ', p)
                    varna = mpimg.imread(p)
                    # ax1[0].set_title('Ground Truth')
                    # print(i, cols, multiplicator, sub)
                    cur_col = i - multiplicator * sub
                    # print(cur_col)
                    if cur_col >= cols:
                        sub = cols
                        multiplicator += 1
                        cur_row += 1
                        cur_col = i - multiplicator * sub
                    # print(cur_row, cur_col)
                    if rows > 1:
                        axs[cur_row][cur_col].imshow(varna, aspect=1)
                        if cur_row == 0:
                            axs[cur_row][cur_col].sharey(axs[cur_row][0])
                            axs[cur_row][cur_col].autoscale()
                        axs[cur_row][cur_col].spines['top'].set_visible(False)
                        axs[cur_row][cur_col].spines['right'].set_visible(False)
                        axs[cur_row][cur_col].spines['left'].set_visible(False)
                        axs[cur_row][cur_col].spines['bottom'].set_visible(False)
                        axs[cur_row][cur_col].axis('off')


                    else:
                        axs[cur_col].imshow(varna)

            # if unbalanced:
            #     axs[rows-1][cols-1].spines['top'].set_visible(False)
            #     axs[rows-1][cols-1].spines['right'].set_visible(False)
            #     axs[rows-1][cols-1].spines['left'].set_visible(False)
            #     axs[rows-1][cols-1].spines['bottom'].set_visible(False)
            #     axs[rows-1][cols-1].axis('off')

            fig.set_size_inches(height,width,forward=False)
            plt.subplots_adjust(hspace=-0.2, wspace=0.001)  # previous hspace = 0.001 or 0.0001
            # plt.rc(usetex=True)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            print('### Save figure to', outpath.resolve())
            # plt.show()
            fig.savefig(outpath, format='svg', dpi=1000, bbox_inches='tight')  #  dpi=300, bbox_inches='tight') #  -> incompatible with constrained layout
            plt.close('all')

    def plot_matrices(self,
                      pred_pairs,
                      true_pairs,
                      length,
                      show: bool = True,
                      out_dir = 'plots',
                      id = 'example'
                      ):
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        pred = pairs2mat(pred_pairs, length)
        true = pairs2mat(true_pairs, length)

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(true)
        ax[1].imshow(pred)

        ax[0].set_title('True Matrix')
        ax[1].set_title('Predicted Matrix')

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if show:
            plt.show()
        outpath = Path(out_dir, f"matrix_representation_{id}")
        print('### Save figure to', outpath.resolve())
        fig.savefig(outpath, format='png', dpi=300, bbox_inches='tight')  #  dpi=300, bbox_inches='tight') #  -> incompatible with constrained layout
        plt.close('all')



    def plot(self, type, *args, **kwargs):
        pass

    def performance_plot_single_dir(self,
                                    results_dir: str,
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
                                    title=None,
                                    labels=None,
                                    file_list=None,
                                    ):
        fig = go.Figure()
        if file_list is None:
            file_list = list(Path(results_dir).glob('*.plk'))
        ranking = []
        if labels is not None:
            assert len(labels) == len(file_list)
            file_label = [(f, l) for f, l in zip(file_list, labels)]
        else:
            file_label = [(f, f.stem) for f in file_list]
        for p, l in file_label:

            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")

            df = pd.read_pickle(p)

            if min_length is not None:
                df = df[df['sequence'].apply(lambda x: min_length <= len(x))]
            if max_length is not None:
                df = df[df['sequence'].apply(lambda x: len(x) <= max_length)]
            if not nc:
                df = df[df['has_nc'] == False]
            if not pks:
                df = df[df['has_pk'] == False]
            if not multiplets:
                df = df[df['has_multiplet'] == False]
            if pk_only:
                df = df[df['has_pk'] == True]
            if multiplets_only:
                df = df[df['has_multiplet'] == True]
            if nc_only:
                df = df[df['has_nc'] == True]

            df = pd.DataFrame.from_dict(df['per_sample_metrics'].to_list())
            if metrics is not None:
                apply_metrics = np.unique([m for m in metrics if m in df.columns])
                df = df[apply_metrics]

            if best_three_key is None:
                fig = self.add_trace_to_spider_plot(fig=fig, df=df, name=l)
            else:
                ranking.append((l, df))
        if best_three_key is not None:
            best_three = sorted(ranking, reverse=True, key=lambda x: np.mean(x[1][best_three_key]))
            best_three=best_three[:n_best]

            for d, b in best_three:
                fig = self.add_trace_to_spider_plot(fig=fig, df=b, name=str(d).split('/')[-1])

        if not legend:
            fig.update_layout(showlegend=False)
        if show:
            fig.show()

        if out_dir is not None:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            out_path = Path(out_dir, f"performance_comparison-{'-'.join(['min_len', str(min_length), 'max_len', str(max_length), 'nc', str(nc), 'pks', str(pks), 'multi', str(multiplets), 'nc_only', str(nc_only), 'pk_only', str(pk_only), 'multi_only', str(multiplets_only)])}-{'-'.join(now.split('/')).replace(' ', '-')}.{output_format}")

            # to avoid a bug in plotly with kaleido writing pdf;
            # see https://github.com/plotly/plotly.py/issues/3469 and
            # https://github.com/plotly/Kaleido/issues/122
            # we write figure twice for pdf format...
            if output_format ==  'pdf':
                fig.write_image(out_path)
                time.sleep(2)
            fig.write_image(out_path)


    def compare_performance_plot(self,
                                results_dir: str,
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
                                directory_list=None,
                                title=None,
                                fraction=1.,
                                log=False,
                                radial_axis_range=None,
                                log_radial_axis_range=[-1, 0],
                                legend_only=False,
                                radial_dtick=0.2,
                                ):
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        fig = go.Figure()
        # fig.update_layout(width=width, height=height, polar=dict(radialaxis_range=log_radial_axis_range if log else radial_axis_range, radialaxis_dtick = 0.1))
        width, height = set_size_px(WIDTH_PT, fraction)
        fig.update_layout(height=height,
                          width=width,
                          legend=dict(
                                  # itemclick=False,  # Disable interactivity on legend items
                                  # itemdoubleclick=False,
                                  itemsizing='constant',  # Set the item size to a constant value
                                  itemwidth=LEGEND_ITEM_SIZE,  # Set the desired item size (in pixels)
                                  font=dict(
                                    size= LEGEND_FONTSIZE*fraction,
                                  ),
                          ),
                          #title_automargin=True,
                          # legend_font_size=LEGEND_FONTSIZE*fraction,
                          title_font_size=TITEL_FONTSIZE*fraction,
                          font_size=TICK_FONTSIZE*fraction,
                          # legend=dict(
                        #     font=dict(
                        #               size=50,
                        #               ),
                          # ),
                          polar=dict(
                              radialaxis_tickfont_size=TICK_FONTSIZE*fraction,
                              radialaxis_type="log" if log else None,
                              radialaxis_range=log_radial_axis_range if log else radial_axis_range,
                              # radialaxis_angle = -45,
                              # radialaxis_rangemode = "normal",
                              radialaxis_gridwidth=GRID_WIDTH,
                              radialaxis_dtick = radial_dtick,
                              angularaxis=dict(
                                  tickfont_size=LABEL_FONTSIZE*fraction,
                                  gridwidth=GRID_WIDTH,
                              ))
                          )
        if legend_only:
            fig.update_polars(
               radialaxis_visible=False,
               angularaxis_visible=False,
               angularaxis_showgrid=False,
               radialaxis_showgrid=False
            )
            fig.update_layout(
               polar=dict(
                   bgcolor='rgba(0,0,0,0)'  # Set the background color to transparent
               ),
               showlegend=True,
            )

        latest_file_list = []
        if best_three_key is not None:
            ranking = []
        if directory_list is not None:
            iterator = directory_list
        else:
            iterator = Path(results_dir).iterdir()
        for directory in iterator:
            if directory.is_file():
                continue
            list_of_directory_files = list(Path(directory).glob('*.plk'))
            latest_file = max(list_of_directory_files, key=os.path.getctime)
            df = pd.read_pickle(latest_file)

            if min_length is not None:
                df = df[df['sequence'].apply(lambda x: min_length <= len(x))]
            if max_length is not None:
                df = df[df['sequence'].apply(lambda x: len(x) <= max_length)]
            if not nc:
                df = df[df['has_nc'] == False]
            if not pks:
                df = df[df['has_pk'] == False]
            if not multiplets:
                df = df[df['has_multiplet'] == False]
            if pk_only:
                df = df[df['has_pk'] == True]
            if multiplets_only:
                df = df[df['has_multiplet'] == True]
            if nc_only:
                df = df[df['has_nc'] == True]
            # print(df)
            # print(metrics)
            # print(df.columns)
            df = pd.DataFrame.from_dict(df['per_sample_metrics'].to_list())
            if metrics is not None:
                apply_metrics = np.unique([m for m in metrics if m in df.columns])
                df = df[apply_metrics]
            # else:
            #     df = pd.DataFrame.from_dict(df['per_sample_metrics'].to_list())
            if best_three_key is None:
                fig = self.add_trace_to_spider_plot(fig=fig, df=df, name=str(directory).split('/')[-1], legend_only=legend_only)
            else:
                ranking.append((directory, df))
        if best_three_key is not None:
            best_three = sorted(ranking, reverse=True, key=lambda x: np.mean(x[1][best_three_key]))
            best_three=best_three[:n_best]

            for d, b in best_three:
                fig = self.add_trace_to_spider_plot(fig=fig, df=b, name=str(d).split('/')[-1], legend_only=legend_only)

        if not legend:
            fig.update_layout(showlegend=False)
        if show:
            fig.show()

        if out_dir is not None:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            out_path = Path(out_dir, f"performance_comparison-{'-'.join(['min_len', str(min_length), 'max_len', str(max_length), 'nc', str(nc), 'pks', str(pks), 'multi', str(multiplets), 'nc_only', str(nc_only), 'pk_only', str(pk_only), 'multi_only', str(multiplets_only)])}-{'-'.join(now.split('/')).replace(' ', '-')}.{output_format}")

            # to avoid a bug in plotly with kaleido writing pdf;
            # see https://github.com/plotly/plotly.py/issues/3469 and
            # https://github.com/plotly/Kaleido/issues/122
            # we write figure twice for pdf format...
            if output_format ==  'pdf':
                fig.write_image(out_path)
                time.sleep(2)
            fig.write_image(out_path)

    def per_dataset_performance_comparison_plots(self,
                                                 results_dir: str,
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
                                                 directory_list=None,
                                                 title=None,
                                                 ):
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        fig = go.Figure()
        latest_file_list = []
        dataset_dfs = defaultdict(list)
        if best_three_key is not None:
            ranking = []
        if directory_list is not None:
            iterator = directory_list
        else:
            iterator = Path(results_dir).iterdir()
        for directory in iterator:
            if directory.is_file():
                continue
            list_of_directory_files = list(Path(directory).glob('*.plk'))
            latest_file = max(list_of_directory_files, key=os.path.getctime)
            df = pd.read_pickle(latest_file)

            if min_length is not None:
                df = df[df['sequence'].apply(lambda x: min_length <= len(x))]
            if max_length is not None:
                df = df[df['sequence'].apply(lambda x: len(x) <= max_length)]
            if not nc:
                df = df[~df['has_nc']]
            if not pks:
                df = df[~df['has_pk']]
            if not multiplets:
                df = df[~df['has_multiplet']]
            if pk_only:
                df = df[df['has_pk'] == True]
            if multiplets_only:
                df = df[df['has_multiplet'] == True]
            if nc_only:
                df = df[df['has_nc'] == True]

            for dataset, d in df.groupby('origin'):
                if metrics is not None:
                    apply_metrics = np.unique([m for m in metrics if m in d.columns])
                    d = d[apply_metrics]
                else:
                    d = pd.DataFrame.from_dict(d['per_sample_metrics'].to_list())
                dataset_dfs[dataset].append((directory, d))
        has_legend = False
        for dataset, df_list in dataset_dfs.items():
            for directory, d in df_list:
                if best_three_key is None:
                    fig = self.add_trace_to_spider_plot(fig=fig, df=d, name=str(directory).split('/')[-1])
                else:
                    ranking.append((directory, d))
            if best_three_key is not None:
                best_three = sorted(ranking, reverse=True, key=lambda x: np.mean(x[1][best_three_key]))
                best_three=best_three[:n_best]
                for d, b in best_three:
                    if not has_legend:
                        fig = self.add_trace_to_spider_plot(fig=fig, df=b, name=str(d).split('/')[-1], show_legend=True)
                    else:
                        fig = self.add_trace_to_spider_plot(fig=fig, df=b, name=str(d).split('/')[-1], show_legend=False)

            if not legend:
                fig.update_layout(showlegend=False)
            if title is not None:
                fig.update_layout(title_text = dataset)
            if show:
                fig.show()

            if out_dir is not None:
                Path(out_dir).mkdir(exist_ok=True, parents=True)
                out_path = Path(out_dir, f"performance_comparison_{dataset}-{'-'.join(['min_len', str(min_length), 'max_len', str(max_length), 'nc', str(nc), 'pks', str(pks), 'multi', str(multiplets), 'nc_only', str(nc_only), 'pk_only', str(pk_only), 'multi_only', str(multiplets_only)])}-{'-'.join(now.split('/')).replace(' ', '-')}.{output_format}")

                # to avoid a bug in plotly with kaleido writing pdf;
                # see https://github.com/plotly/plotly.py/issues/3469 and
                # https://github.com/plotly/Kaleido/issues/122
                # we write figure twice for pdf format...
                if output_format ==  'pdf':
                    fig.write_image(out_path)
                    time.sleep(2)
                fig.write_image(out_path)
            fig.data = []


    def add_trace_to_spider_plot(self,
                                 fig: go.Figure,
                                 df: pd.DataFrame,
                                 name: str,
                                 fill=None,  # 'toself',  # 'tonext',  # 'toself',  #None  # 'toself',
                                 opacity=1.0,
                                 show_legend=True,
                                 legend_only=False,
                                 ):
        y = [np.mean(df[k]) if not 'solved' in k or not 'valid' in k else np.sum(df[k]) / len(df) for k in df.columns]   # df.iloc[0].tolist()
        y += [y[0]]
        x = [c for c in df.columns]
        x += [x[0]]
        if legend_only:
            line_dict=dict(
              width=LINE_WIDTH,  # Set the linewidth
              # color='rgba(0,0,0,0)',
            )
            visible = 'legendonly'
        else:
            line_dict=dict(
              width=LINE_WIDTH,  # Set the linewidth
            )
            visible = True
        fig.add_trace(go.Scatterpolar(
            r=y,
            theta=x,
            fill=fill,
            opacity=opacity,
            line=line_dict,
            # fillcolor='rgba(26,150,65,0)',  #
            showlegend=show_legend,
            visible=visible,
            name=name))
        return fig

    def analyze_runtime_latest_runs(self,
                                    results_dir: str,
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
                                    ):

        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        fig = go.Figure()
        latest_file_list = []
        if best_three_key is not None:
            ranking = []
        if file_list is not None:
            iterator = file_list
        else:
            iterator = list(Path(results_dir).glob('**/*.plk'))
        print(iterator)
        x = []
        y = []
        for fi in iterator:
            # if directory.is_file():
            #     continue
            # list_of_directory_files = list(Path(directory).glob('*.plk'))
            # latest_file = max(list_of_directory_files, key=os.path.getctime)
            df = pd.read_pickle(fi)

            if min_length is not None:
                df = df[df['sequence'].apply(lambda x: min_length <= len(x))]
            if max_length is not None:
                df = df[df['sequence'].apply(lambda x: len(x) <= max_length)]
            if not nc:
                df = df[~df['has_nc']]
            if not pks:
                df = df[~df['has_pk']]
            if not multiplets:
                df = df[~df['has_multiplet']]
            if pk_only:
                df = df[df['has_pk'] == True]
            if multiplets_only:
                df = df[df['has_multiplet'] == True]
            if nc_only:
                df = df[df['has_nc'] == True]

            df = pd.DataFrame.from_dict(df['per_sample_metrics'].to_list())
            df = df[['time']]
            x = [str(fi).split('/')[-1].split('_')[0]]

            # x.append(str(fi).split('/')[-1].split('_')[0])
            y = [np.mean(df['time'].to_list())]
            # y.append(np.mean(df['time'].to_list()))
            # print(x)
            # print(y)
            fig = self.add_trace_to_bar_plot(fig=fig, x=x, y=y, name=x[0])
        # fig = self.add_trace_to_bar_plot(fig=fig, x=x, y=y)

        if not legend:
            fig.update_layout(showlegend=False)
        if show:
            fig.show()

        if out_dir is not None:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            out_path = Path(out_dir, f"performance_comparison-{'-'.join(['min_len', str(min_length), 'max_len', str(max_length), 'nc', str(nc), 'pks', str(pks), 'multi', str(multiplets), 'nc_only', str(nc_only), 'pk_only', str(pk_only), 'multi_only', str(multiplets_only)])}-{'-'.join(now.split('/')).replace(' ', '-')}.{output_format}")

            # to avoid a bug in plotly with kaleido writing pdf;
            # see https://github.com/plotly/plotly.py/issues/3469 and
            # https://github.com/plotly/Kaleido/issues/122
            # we write figure twice for pdf format...
            if output_format ==  'pdf':
                fig.write_image(out_path)
                time.sleep(2)
            fig.write_image(out_path)

    def add_trace_to_bar_plot(self,
                              fig,
                              x,
                              y,
                              name,
                              ):
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=name,
            marker=dict(color='darkblue'),  # 'darkcyan')
            # color='blue',
            ))
        return fig
        # databases = ['MySQL', 'MongoDB', 'Elasticsearch', 'Redis', 'SQLite']
        # usage = [80, 75, 42, 78, 60]
        # fig = go.Figure(data=go.Bar(
        #     x=databases,
        #     y=usage
        # ))
        ### for different layouts:
        # fig.update_layout(barmode='stack')
        # fig.update_layout(barmode='group')
        # fig.update_layout(barmode='stack', xaxis={'categoryorder': "total descending"})
        # fig.update_layout(barmode='stack', xaxis={'categoryorder': "total ascending"})
        ###
        # fig.show()

    def histo_dataset_comparison(self,
                                per_sample_stats,  # : List[pd.DataFrame],
                                param='length',
                                bin_size=20,
                                labels=None,
                                show=True,
                                out_dir='plots',
                                output_format='pdf',
                                show_rug=True,
                                colors=None,
                                show_curve=True,
                                ):
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        labels = labels if labels is not None else list(range(len(per_sample_stats)))
        fig = ff.create_distplot([df[param] for df in per_sample_stats],
                                 labels,
                                 bin_size=bin_size,
                                 show_rug=show_rug,
                                 show_curve=show_curve,
                                 )
        if show:
            fig.show()
        if out_dir is not None:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            out_path = Path(out_dir, f"dataset_comparison-{'-'.join(labels)}-{'-'.join(now.split('/')).replace(' ', '-')}.{output_format}")

            # to avoid a bug in plotly with kaleido writing pdf;
            # see https://github.com/plotly/plotly.py/issues/3469 and
            # https://github.com/plotly/Kaleido/issues/122
            # we write figure twice for pdf format...
            if output_format ==  'pdf':
                fig.write_image(out_path)
                time.sleep(2)
            fig.write_image(out_path)


    def histo_dataset_comparison_express(self,
                                    per_sample_stats,  # : List[pd.DataFrame],
                                    param='length',
                                    # bin_size=20,
                                    labels=None,
                                    show=True,
                                    out_dir='plots',
                                    output_format='pdf',
                                    marginal='violin',
                                    histnorm='probability density',  # 'density',  #'probability',  # 'probability density',
                                    opacity=1.0,
                                    color_discrete_sequence=plotly_colors['Pastel1'],  #'Pastel1', 'T10'
                                    nbins=100,
                                    width=1920,
                                    height=1080,
                                    no_bg=True,
                                    fraction=1.0,
                                    grid_color='lightgray',  # 'darkslategray',  #
                                    zero_color='black',
                                    axis_color='black',
                                    grid_size=1,
                                    zero_size=1,
                                    axis_size=2,
                                         ):

        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        labels = labels if labels is not None else list(range(len(per_sample_stats)))
        # data = [df[param].to_list() for df in per_sample_stats]
        df_list = []
        for i, df in enumerate(per_sample_stats):
            df.loc[:, 'Dataset'] = labels[i]
            df_list.append(df)
        df = pd.concat(df_list)

        width, height = set_size_px(WIDTH_PT, fraction)

        fig = px.histogram(df,
                           labels=labels,
                           nbins=nbins,
                           x=param,
                           color='Dataset',
                           histnorm=histnorm,
                           marginal=marginal,  # 'box',  # 'violin',
                           opacity=opacity,
                           color_discrete_sequence=color_discrete_sequence,
                           width=width,
                           height=height,
                           )
        fig.update_layout(
                          legend=dict(
                                  # itemclick=False,  # Disable interactivity on legend items
                                  # itemdoubleclick=False,
                                  itemsizing='constant',  # Set the item size to a constant value
                                  itemwidth=LEGEND_ITEM_SIZE,  # Set the desired item size (in pixels)
                                  font=dict(
                                    size= LEGEND_FONTSIZE*fraction,
                                  ),
                          ),
                          #title_automargin=True,
                          # legend_font_size=LEGEND_FONTSIZE*fraction,
                          title_font_size=TITEL_FONTSIZE*fraction,
                          font_size=TICK_FONTSIZE*fraction,
                          # legend=dict(
                        #     font=dict(
                        #               size=50,
                        #               ),
                          )

        if no_bg:
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            fig.update_yaxes(showgrid=True,
                             gridwidth=grid_size,
                             gridcolor=grid_color,
                             zeroline=True,
                             zerolinewidth=zero_size,
                             zerolinecolor=zero_color,
                             showline=True,
                             linewidth=axis_size,
                             linecolor=axis_color,)

        if show:
            fig.show()
        if out_dir is not None:
            Path(out_dir).mkdir(exist_ok=True, parents=True)
            out_path = Path(out_dir, f"dataset_comparison-{'-'.join(labels)}-{'-'.join(now.split('/')).replace(' ', '-')}.{output_format}")

            # to avoid a bug in plotly with kaleido writing pdf;
            # see https://github.com/plotly/plotly.py/issues/3469 and
            # https://github.com/plotly/Kaleido/issues/122
            # we write figure twice for pdf format...
            if output_format ==  'pdf':
                fig.write_image(out_path)
                time.sleep(2)
            fig.write_image(out_path)



    def bar_plot(self):
        pass

    def scatter_plot(self):
        pass

    def line_plot(self):
        pass



class RNAStatistics():
    def __init__(self, dataset):  # : 'RnaBench.core.datasets.RnaDataSet'
        self.data = dataset

    def get_detailed_statistics(self):
        self.detailed_statistics = defaultdict(list)
        self.per_sample_stats = []

        nucs = ''

        for rna in self.data:
            per_sample_res = {}
            pair_numbers = defaultdict(list)
            seq = ''.join(rna.sequence)
            nucs += seq  #  ''.join(seq)


            self.detailed_statistics['length'].append(len(seq))
            per_sample_res['length'] = len(seq)

            pairs = rna.pairs  #  [(p1, p2, pk) for p1, p2, pk in zip(rna.pos1id, rna.pos2id, rna.pk)]
            per_sample_res['num_pairs'] = len(pairs)


            self.detailed_statistics['all_pairs'] += pairs

            for p in pairs:
                pair_numbers[p[0]].append(p[1])
                pair_numbers[p[1]].append(p[0])
                type = sorted([seq[p[0]], seq[p[1]]])
                if not '-'.join(type) in per_sample_res.keys():
                    per_sample_res['-'.join(type)] = [(p[0], p[1])]
                else:
                    per_sample_res['-'.join(type)].append((p[0], p[1]))
                self.detailed_statistics['-'.join(type)].append(1)
                if '-'.join(type) not in ['G-U', 'A-U', 'C-G']:
                    self.detailed_statistics['nc_pairs'].append(1)
                # self.detailed_statistics['page'].append(p[2])

            wc, wobble, nc = get_pair_types(seq, pairs)
            self.detailed_statistics['WC'].append(wc)
            self.detailed_statistics['Wobble'].append(wobble)
            self.detailed_statistics['NC'].append(nc)
            per_sample_res['WC'] = wc
            per_sample_res['Wobble'] = wobble
            per_sample_res['NC'] = nc

            pk_pairs = [(p1, p2) for (p1, p2, pk) in pairs if pk > 0]
            self.detailed_statistics['PK'].append(pk_pairs)
            per_sample_res['PKs'] = pk_pairs

            if pk_pairs:
                self.detailed_statistics['pk_sample'].append(True)
            else:
                self.detailed_statistics['pk_sample'].append(False)

            multiplets = []

            for k, v in pair_numbers.items():
                # print(multiplets)
                if len(v) > 1:
                    for p in v:
                        # print(p, v)
                        multiplets.append(tuple(sorted([k, p])))
            # print(multiplets, type(multiplets), set(multiplets))
            multiplets = list(set(multiplets))

            per_sample_res['multiplets'] = multiplets

            self.detailed_statistics['multiplets'].append(multiplets)

            if multiplets:
                self.detailed_statistics['multiplet_sample'].append(True)
            else:
                self.detailed_statistics['multiplet_sample'].append(False)

            per_sample_res['A'] = seq.upper().count('A')
            per_sample_res['C'] = seq.upper().count('C')
            per_sample_res['G'] = seq.upper().count('G')
            per_sample_res['U'] = seq.upper().count('U')

            per_sample_res['Id'] = rna.id
            self.per_sample_stats.append(per_sample_res)

        self.detailed_statistics['A'] = nucs.upper().count('A')
        self.detailed_statistics['C'] = nucs.upper().count('C')
        self.detailed_statistics['G'] = nucs.upper().count('G')
        self.detailed_statistics['U'] = nucs.upper().count('U')

        # return self.detailed_statistics

    def get_dataset_statistics(self):
        self.get_detailed_statistics()
        self.dataset_summary = {
          'num_samples': len(self.data),
          'min_length': self.data.data['length'].min(),
          'max_length': self.data.data['length'].max(),
          'mean_length': self.data.data['length'].mean(),
          'median_length': self.data.data['length'].median(),
          'num_pk_samples': self.data.data['has_pk'].sum(),
          'num_multiplet_samples': self.data.data['has_multiplet'].sum(),
          'num_nc_samples': self.data.data['has_nc'].sum(),
          'total_num_pairs': len(self.detailed_statistics['all_pairs']),
        }

        self.dataset_summary.update({
          'A-ratio': self.detailed_statistics['A'] / self.dataset_summary['num_samples'],
          'C-ratio': self.detailed_statistics['C'] / self.dataset_summary['num_samples'],
          'G-ratio': self.detailed_statistics['G'] / self.dataset_summary['num_samples'],
          'U-ratio': self.detailed_statistics['U'] / self.dataset_summary['num_samples'],
        })

        self.dataset_summary.update({
          'GC-pair-ratio': len(self.detailed_statistics['C-G']) / len(self.detailed_statistics['all_pairs']),
          'AU-pair-ratio': len(self.detailed_statistics['A-U']) / len(self.detailed_statistics['all_pairs']),
          'GU-pair-ratio': len(self.detailed_statistics['G-U']) / len(self.detailed_statistics['all_pairs']),
          'nc-pair-ratio': len(self.detailed_statistics['nc_pairs']) / len(self.detailed_statistics['all_pairs']),
        })


    def performance_steatistics(self):
        pass

    @property
    def per_sample_stats_df(self):
        return pd.DataFrame.from_dict(self.per_sample_stats)


if __name__ == '__main__':
    from RnaBench.lib.datasets import RnaDataset
    # benchmark = 'intra_family'
    # benchmark = 'inter_family'
    # benchmark = 'biophysical_model'
    # benchmark = 'inverse_rna_folding'
    benchmark = 'constrained_design'
    labels = []
    df_list = []
    bin_size = 5

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



    # stats2.get_dataset_statistics()
    # stats3.get_dataset_statistics()
    # print(stats.dataset_summary)
    vis = RnaVisualizer()
    # print(stats.per_sample_stats_df)
    # vis.histo_dataset_comparison(df_list,
    #                              labels=labels,
    #                              bin_size=bin_size)
    vis.histo_dataset_comparison_express(df_list,
                                         labels=labels,
                                         )
                                         # bin_size=bin_size)

    # print(stats.detailed_statistics)





