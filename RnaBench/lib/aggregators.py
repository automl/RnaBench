import numpy as np
import pandas as pd

from RnaBench.lib.structural_motifs import Stem, ILoop, Hairpin, MLoop


def forgi_stem_feat_aggregator(bg, func, aggregation, rna_id, reduction="sum"):
    stems_feats = []
    for stem_id in bg.sorted_stem_iterator():
        # create stem object
        # call the function on the stem object
        s = Stem(rna_id, stem_id, length=bg.get_length(stem_id),
                 seq=bg.get_define_seq_str(stem_id),
                 rng_5p=bg.get_side_nucleotides(stem_id, 0),
                 rng_3p=bg.get_side_nucleotides(stem_id, 1))

        stems_feats.append(func(s))

    stems_array = np.array(stems_feats)
    if stems_array.shape[0] == 0:
        stems_array = np.array([func(None)])

    if aggregation == 'strand':
        stems_stats = np.mean(stems_array, axis=0) if reduction == 'mean' else np.sum(stems_array, axis=0)
        return stems_stats

    stems_df = pd.DataFrame(stems_array)
    stems_df["rna_id"] = rna_id
    stems_df["type"] = 's'
    stems_df["motif_idx"] = np.arange(len(stems_feats)) if len(stems_feats) != 0 else 0

    return stems_df


def forgi_iloop_feat_aggregator(bg, func, aggregation, rna_id, reduction="sum"):
    iloops_feats = []
    for iloop_id in bg.iloop_iterator():

        iloop_seq = bg.get_define_seq_str(iloop_id)
        if len(bg.get_define_seq_str(iloop_id)) == 4:
            iloop_seq = [iloop_seq[0], iloop_seq[2]]

        nuc_ids = bg.elements_to_nucleotides([iloop_id])
        cp_5p_rng = (nuc_ids[0], nuc_ids[len(iloop_seq[0]) - 1]) if len(iloop_seq[0]) != 0 else (None, None)
        cp_3p_rng = (nuc_ids[len(iloop_seq[0])], nuc_ids[-1]) if len(iloop_seq[1]) != 0 else (None, None)
        start_idx, end_idx, _, _ = bg.get_flanking_handles(iloop_id)
        cp_5p_nuc = bg.seq[start_idx], bg.seq[end_idx]
        start_idx, end_idx, _, _ = bg.get_flanking_handles(iloop_id, side=1)
        cp_3p_nuc = bg.seq[start_idx], bg.seq[end_idx]
        il = ILoop(rna_id, iloop_id,
                   seq=bg.get_define_seq_str(iloop_id),
                   cp_5p=cp_5p_rng,
                   cp_3p=cp_3p_rng,
                   cp_5p_nuc=cp_5p_nuc,
                   cp_3p_nuc=cp_3p_nuc)

        iloops_feats.append(func(il))

    iloops_array = np.array(iloops_feats)

    if iloops_array.shape[0] == 0:
        iloops_array = np.array([func(None)])

    if aggregation == 'strand':
        iloops_stats = np.mean(iloops_array, axis=0) if reduction == 'mean' else np.sum(iloops_array, axis=0)
        return iloops_stats

    iloops_df = pd.DataFrame(iloops_array)
    iloops_df["rna_id"] = rna_id
    iloops_df["type"] = 'i'
    iloops_df["motif_idx"] = np.arange(len(iloops_feats)) if len(iloops_feats) != 0 else 0
    return iloops_df


def forgi_hairpin_feat_aggregator(bg, func, aggregation, rna_id, reduction="sum"):
    #TODO fix length 1 bug
    h_feats = []
    for h_id in bg.hloop_iterator():
        # create stem object
        # call the function on the   stem object
        h_seq = bg.get_define_seq_str(h_id)
        nuc_ids = bg.elements_to_nucleotides([h_id])
        rng = (nuc_ids[0], nuc_ids[-1])
        start_idx, end_idx, _, _ = bg.get_flanking_handles(h_id)
        nuc = bg.seq[start_idx], bg.seq[end_idx]
        hp = Hairpin(rna_id, h_id, seq=h_seq[0], rng=rng, cp=nuc)
        h_feats.append(func(hp))

    h_array = np.array(h_feats)
    if h_array.shape[0] == 0:
        h_array = np.array([func(None)])

    if aggregation == 'strand':
        stems_stats = np.mean(h_array, axis=0) if reduction == 'mean' else np.sum(h_array, axis=0)
        return stems_stats

    hairpins_df = pd.DataFrame(h_array)
    hairpins_df["rna_id"] = rna_id
    hairpins_df["type"] = 'h'
    hairpins_df["motif_idx"] = np.arange(len(h_feats)) if len(h_feats) != 0 else 0

    return hairpins_df


def bprna_stem_feat_aggregator(motif_df, func, aggregation, rna_id=None, reduction="sum"):
    stem_df_c = motif_df[motif_df['type'] == 's'].copy()
    if stem_df_c.shape[0] > 0:
        def info_list_to_stem(df):
            stem = Stem(df.rna_id, df.motif_idx)
            stem.from_bprna_list(df["info"])
            return stem

        stem_df_c['stem_obj'] = stem_df_c.apply(info_list_to_stem, axis=1)
        stems_feat_df = stem_df_c.apply(lambda x: func(x["stem_obj"]),
                                        result_type="expand",
                                        axis=1)

        stems_array = np.array(stems_feat_df.values)

    else:
        stems_array = np.array([func(None)])

    if aggregation == 'strand':
        stems_stats = np.nanmean(stems_array, axis=0) if reduction == 'mean' else np.nansum(stems_array, axis=0)
        return stems_stats

    stems_feat_df = pd.DataFrame(stems_array)

    return pd.concat([stem_df_c.loc[:, ['rna_id', 'motif_idx', 'type']], stems_feat_df], axis=1)


def bprna_iloop_feat_aggregator(motif_df, func, aggregation, rna_id=None, reduction="sum"):
    iloop_df_c = motif_df[motif_df['type'] == 'i'].copy()
    if iloop_df_c.shape[0] > 0:
        new_columns = iloop_df_c.loc[:, "motif_idx"].str.split(pat="\.+", expand=True).astype(float)
        iloop_df_c[["motif_idx", "five_three"]] = new_columns
        info = iloop_df_c.groupby("motif_idx").apply(lambda x: [x.loc[x["five_three"] == 1.0, "info"].values[0],
                                                                x.loc[x["five_three"] == 2.0, "info"].values[0]])
        iloop_df_c.drop_duplicates(subset=['motif_idx'], inplace=True)
        iloop_df_c.loc[:, "info"] = info.values

        def info_list_to_iloop(df):
            iloop = ILoop(df.rna_id, df.motif_idx)
            iloop.from_bprna_list(df["info"])
            return iloop

        iloop_df_c['iloop_obj'] = iloop_df_c.apply(info_list_to_iloop, axis=1)
        iloops_feat_df = iloop_df_c.apply(lambda x: func(x["iloop_obj"]),
                                          result_type="expand",
                                          axis=1)

        iloops_array = np.array(iloops_feat_df.values)

    else:
        iloops_array = np.array([func(None)])

    # if iloops_array.shape[0] == 0:
    #    iloops_array = np.array([func(None)])

    if aggregation == 'strand':
        iloops_stats = np.nanmean(iloops_array, axis=0) if reduction == 'mean' else np.nansum(iloops_array, axis=0)
        return iloops_stats

    iloops_feat_df = pd.DataFrame(iloops_array)

    return pd.concat([iloop_df_c.loc[:, ['rna_id', 'motif_idx', 'type']], iloops_feat_df], axis=1)


def bprna_hairpin_feat_aggregator(motif_df, func, aggregation, rna_id=None, reduction="sum"):
    hairpin_df_c = motif_df[motif_df['type'] == 'h'].copy()
    if hairpin_df_c.shape[0] > 0:

        def info_list_to_hairpint(df):
            iloop = Hairpin(df.rna_id, df.motif_idx)
            iloop.from_bprna_list(df["info"])
            return iloop

        hairpin_df_c['hairpin_obj'] = hairpin_df_c.apply(info_list_to_hairpint, axis=1)
        hairpin_feat_df = hairpin_df_c.apply(lambda x: func(x["hairpin_obj"]),
                                             result_type="expand",
                                             axis=1)

        hairpins_array = np.array(hairpin_feat_df.values)

    else:
        hairpins_array = np.array([func(None)])
        # return hairpins_array

    if aggregation == 'strand':
        hairpins_stats = np.nanmean(hairpins_array, axis=0) if reduction == 'mean' else np.nansum(hairpins_array, axis=0)
        return hairpins_stats

    hairpins_feat_df = pd.DataFrame(hairpins_array)

    return pd.concat([hairpin_df_c.loc[:, ['rna_id', 'motif_idx', 'type']], hairpins_feat_df], axis=1)


def bprna_mloop_feat_aggregator(motif_df, func, aggregation, rna_id=None, reduction="sum"):
    mloop_df_c = motif_df[motif_df['type'] == 'm'].copy()
    if mloop_df_c.shape[0] > 0:
        new_columns = mloop_df_c.loc[:, "motif_idx"].str.split(pat="\.+", expand=True).astype(float)
        mloop_df_c[["motif_idx", "five_three"]] = new_columns

        def extract_info(x):
            info = []
            strands = x["five_three"].unique()
            strands.sort()
            for strand in strands:
                info.append(x.loc[x["five_three"] == strand, "info"].values[0])
            return info

        info = mloop_df_c.groupby("motif_idx").apply(lambda x: extract_info(x))
        mloop_df_c.drop_duplicates(subset=['motif_idx'], inplace=True)
        mloop_df_c.loc[:, "info"] = info.values

        def info_list_to_mloop(df):
            iloop = MLoop(df.rna_id, df.motif_idx)
            iloop.from_bprna_list(df["info"])
            return iloop

        mloop_df_c['mloop_obj'] = mloop_df_c.apply(info_list_to_mloop, axis=1)
        mloops_feat_df = mloop_df_c.apply(lambda x: func(x["mloop_obj"]),
                                          result_type="expand",
                                          axis=1)

        mloops_array = np.array(mloops_feat_df.values)

    else:
        mloops_array = np.array([func(None)])

    if aggregation == 'strand':
        mloops_stats = np.nanmean(mloops_array, axis=0) if reduction == 'mean' else np.nansum(mloops_array, axis=0)
        return mloops_stats

    mloops_feat_df = pd.DataFrame(mloops_array)

    return pd.concat([mloop_df_c.loc[:, ['rna_id', 'motif_idx', 'type']], mloops_feat_df], axis=1)
