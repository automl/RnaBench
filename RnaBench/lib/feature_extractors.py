import os
import shutil
import subprocess
from datetime import datetime

import forgi.graph.bulge_graph as fgb
import regex as re

from RnaBench.lib.aggregators import *
from RnaBench.lib.utils import pairs2db

default_struct_funcs = {
    "s": lambda x: [float(len(x))] if x is not None else [np.NaN],
    "h": lambda x: [float(len(x))] if x is not None else [np.NaN],
    "i": lambda x: list(x.size()) if x is not None else [np.NaN, np.NaN],
    "m": lambda x: list(sum(list(x.size()))/len(list(x.size())), x._cps[0][0]) if x is not None else [np.NaN, np.NAN]
}

feat_aggregators = {
    "forgi_s": forgi_stem_feat_aggregator,
    "forgi_i": forgi_iloop_feat_aggregator,
    "forgi_h": forgi_hairpin_feat_aggregator,
    "bprna_s": bprna_stem_feat_aggregator,
    "bprna_i": bprna_iloop_feat_aggregator,
    "bprna_h": bprna_hairpin_feat_aggregator,
    "bprna_m": bprna_mloop_feat_aggregator,
}


# def stem_stats(motif_df):
#     stem_df_c = motif_list[''].copy()
#     new_columns = stem_df_c.loc[:, "5_pos"].str.split(pat="\.+", expand=True).astype(float)
#     stem_df_c[["5_start", "5_end"]] = new_columns
#     new_columns = stem_df_c.loc[:, "3_pos"].str.split(pat="\.+", expand=True).astype(float)
#     stem_df_c[["3_start", "3_end"]] = new_columns
#     stem_df_c["stem_len"] = stem_df_c["5_end"] - stem_df_c["5_start"]
#     # stem_df_c.drop(columns = ["5_pos", "3_pos", 'stem_id' ], inplace=True)
#     # print(stem_df_c.head())
#     stem_df_c = stem_df_c
#     stem_stats_df = stem_df_c.describe().fillna(0)
#     return np.array(stem_stats_df['stem_len'].values)


def forgi_bg(datapoint):
    #check type of datapoint.sequence and datapoint.structure
    #structure = datapoint.structure if isinstance(datapoint.structure, str) else "".join(datapoint.structure)
    #check if sequence is in the datapoint else create from pairs

    if "sequence" in datapoint.index:
        #create from pairs
        sequence = datapoint.sequence if isinstance(datapoint.sequence, str) else "".join(datapoint.sequence)
        # sequence = datapoint.sequence if isinstance(datapoint.sequence, str) else "".join(datapoint.sequence)

    if "structure" in datapoint.index:
        structure = datapoint.structure if isinstance(datapoint.structure, str) else "".join(datapoint.structure)
    elif "pairs" in datapoint.index or "predicted_pairs" in datapoint.index:
        structure = pairs2db(datapoint.pairs if "pairs" in datapoint.index else datapoint.predicted_pairs, sequence)

    # print(datapoint.pairs)
    # print(structure)
    #sequence = datapoint.sequence if isinstance(datapoint.sequence, str) else "".join(datapoint.sequence)
    return fgb.BulgeGraph.from_dotbracket(structure, sequence, remove_pseudoknots=False)


def bprna_motifs(datapoint):
    # This function takes the datapoint consisting of the sequence, the structure and other features
    # and produces comprehensive stats about the
    # structural motifs (hairpins, stems, pseudoknot, etc ...)

    # create the dbn file to use bprna
    None if os.path.isdir('./tmp_data') else os.mkdir('./tmp_data')
    date = datetime.now().strftime("%Y_%m_%d-%I%M%f")
    lines = [datapoint.sequence, "\n", datapoint.structure]
    with open(f'./tmp_data/rna_{date}.dbn', 'w') as f:
        f.writelines(lines)

    subprocess.call(["perl", "../bpRNA/bpRNA.pl", f'rna_{date}.dbn'], cwd='./tmp_data/')
    lookup = "S1 "
    lines = [line.strip('\n') for line in open(f'./tmp_data/rna_{date}.st')]
    start_line = 7
    page_number = 1
    for i, line in enumerate(lines):
        if line.startswith("#PageNumber: "):
            page_number = int(re.findall(r'\d+', line)[0])

        if line.startswith(lookup):
            start_line = i
            break

    result = [line.strip('\n') for line in open(f'./tmp_data/rna_{date}.st')][start_line:]
    motif_raw_list = [[re.sub(r"[\W\d_]+$", "", r.split()[0]),
                       re.findall(r"[\W\d_]+$", r.split()[0])[0],
                       r.split()[1:]] for r in result]

    motif_raw_df = pd.DataFrame(motif_raw_list, columns=['type', 'motif_idx', 'info'])
    motif_raw_df["rna_id"] = datapoint.id if "id" in datapoint else datapoint.Id
    motif_raw_df["page_number"] = page_number
    shutil.rmtree('./tmp_data/')
    return motif_raw_df


class FeaturesExtractor:
    """
    Class that defines a feature extractor, e.g. extracting structural motifs
    """

    def __init__(
            self,
            aggregation_level='strand'
    ):
        self._aggregation_level = aggregation_level  # could be per RNA or across the whole dataset

    def __call__(self, data):
        return self.extract(data)

    def extract(self, datapoint):
        raise NotImplementedError('subclasses must override extract()!')

    @property
    def aggregation_level(self):
        return self._aggregation_level


class StructuralMotifs(FeaturesExtractor):
    """
    Structural motifs extractor
    """

    def __init__(
            self,
            aggregation_level='strand',
            source='forgi',
            sec_struct_feats=("s", "h", "i"),
            type="normal",
            custom_struct_funcs={},
            custom_funcs={}
    ):
        """

        :param aggregation_level: (str) whether to aggregate on dataset level directly or
         carry out aggregation on both levels
        :param source: (str) whether to use forgi or bprna for extracting structural
        features
        :param sec_struct_feats: (list) list of the features to extract its data,
        there could some use of general features
        :param type: (str) the type of the distrubtion to model this feature
        :param custom_struct_funcs: (dict) mapping from structure key or new feat to
        :param custom_funcs: (dict) mapping from new feat that can be a function or one
            or more struct motifs, this needs on iterator on its own.
        """
        super().__init__(aggregation_level)

        self.source = source
        self.sec_struct_feats = sec_struct_feats
        self.custom_struct_funcs = custom_struct_funcs
        self.custom_funcs = custom_funcs
        self.type = type
        self.feat_names = []
        for feat in self.sec_struct_feats:
            func = self.custom_struct_funcs[feat] \
                if feat in self.custom_struct_funcs.keys() else default_struct_funcs[feat]
            x = func(None)
            self.feat_names.extend([f"{feat}_{i}" for i in range(len(x))])

    def extract(self, datapoint):
        feat_list = []
        if self.source == 'forgi':
            rna_motifs = forgi_bg(datapoint)
        elif self.source == "bprna":
            rna_motifs = bprna_motifs(datapoint)
            rna_motifs['type'] = rna_motifs['type'].str.lower()

        else:
            raise NotImplementedError('other sources for structural motifs are not supported!')

        # (TODO) we can add source per feature and switch
        for feat in self.sec_struct_feats:

            func = self.custom_struct_funcs[feat] \
                if feat in self.custom_struct_funcs.keys() else default_struct_funcs[feat]

            feat_values = feat_aggregators[f'{self.source}_{feat}'](
                rna_motifs,
                func,
                self.aggregation_level,
                datapoint.id if "id" in datapoint else datapoint.Id) #TODO determine id to use

            feat_list.append(feat_values)

        if self.aggregation_level == 'strand':
            if len(feat_list) == 1:
                return feat_list[0]

            else:
                return np.concatenate(feat_list, axis=-1)

        elif  self.aggregation_level == "motif":
            return pd.concat(feat_list)

        else:
            feat_df =  pd.concat(feat_list)
            grouped = feat_df.groupby(['rna_id', 'type']).agg(list).reset_index()
            grouped = grouped.drop(['motif_idx', 'rna_id'], axis=1)
            grouped = grouped.set_index('type')
            return [grouped.loc[feat_name.split('_')[0], int(feat_name.split('_')[1])] for feat_name in self.feat_names]
