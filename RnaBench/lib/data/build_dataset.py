import pickle
import subprocess
import RNA

import pandas as pd
import numpy as np

from typing import List, Union, Optional
from re import sub
from pathlib import Path
from collections import defaultdict, Counter

from RnaBench.lib.utils import (
db2pairs,
has_nc,
has_multiplet,
has_seq_nc,
not_seq_nc_only,
df2fasta,
)

# from RnaBench.lib.data.structure_annotation import BpRNA
# from RnaBench.lib.data.data_preprocessing import (
# apply_rfam_family_assignment_pipeline,
# apply_sequence_structure_similarity_pipeline,
# )

from RnaBench.lib.data.pre_processing import SequenceSimilarityPipeline
from RnaBench.lib.data.similarity import Infernal, CDHIT
from RnaBench.lib.data.parser.blast_tab_parser import BlastTabParser


NUCS = {
    'T': 'U',
    'P': 'U',
    'R': 'A',  # or 'G'
    'Y': 'C',  # or 'T'
    'M': 'C',  # or 'A'
    'K': 'U',  # or 'G'
    'S': 'C',  # or 'G'
    'W': 'U',  # or 'A'
    'H': 'C',  # or 'A' or 'U'
    'B': 'U',  # or 'G' or 'C'
    'V': 'C',  # or 'G' or 'A'
    'D': 'A',  # or 'G' or 'U'
    'N': 'N',
    'A': 'A',
    'U': 'U',
    'C': 'C',
    'G': 'G',
}

def replace_iupac(seq):
    return [NUCS[s] for s in ''.join(seq).upper()]

def gen_dup_key(row):
    k = ''.join(row['sequence']) + ','.join([str(x) for x in row['pos1id']]) + ','.join([str(x) for x in row['pos2id']]) + ','.join([str(x) for x in row['pk']])
    return k


def sample_synthetic_sequences(cm_path : str,
                               outpath : str,
                               seed : int,
                               n : int = 250,
                               ) -> pd.DataFrame:

    print('### Sample sequences')
    subprocess.call(["cmemit", "-o", outpath, "-N", str(n), "--seed", str(seed), cm_path])

    print('### Process fasta output')
    tmp_out = Path('tmp.fa')
    subprocess.call(['./RnaBench/lib/data/format_fasta.sh', outpath, str(tmp_out.resolve())])

    with open(tmp_out, 'r') as f:
        lines = f.readlines()

    sequences = []
    for i, line in enumerate(lines):
        if line.startswith('>'):
            line = line.rstrip().split('-')
            model = '-'.join(line[:-1])[1:]
            seq_id = line[-1]
            seq = list(lines[i+1].rstrip())
            sequences.append((model, seq_id, seq))
        else:
            continue
    tmp_out.unlink()
    df = pd.DataFrame(sequences, columns=['family', 'Id', 'sequence'])

    df = df[df['sequence'].apply(lambda x: len(x) > 12)]

    print(df)

    # TODO: get more information about families with cmstat and cmfetch
    return df


def resolve_iupac(df):
    df.loc[:, 'sequence'] = df['sequence'].apply(replace_iupac)
    return df

def pairs2pos1(pairs):
    return [i[0] for i in pairs]

def pairs2pos2(pairs):
    return [i[1] for i in pairs]

def pairs2pk(pairs):
    return [i[2] for i in pairs]

def get_task_seq(task, seq, rng, unknown_character='N', seed=0):
    if task == 'inverse_rna_folding':
        task_seq = [unknown_character] * len(seq)
    elif task == 'constrained_inverse_rna_folding':
        task_seq = random_mask(seq, rng)
    else:
        task_seq = seq
    return task_seq

def random_mask(s, rng, replacement_factor=0.2, num_max_replacements=5, unknown_character='N'):
    """
    Replace random regions of string with unknown character.
    Not simple random masking, we want the replacements not to be
    position-wise but region-wise.
    """
    rep_max_len = int(len(s) * replacement_factor)
    if rep_max_len == 0:
        rep_max_len = 1
    for i in range(num_max_replacements):
        rep_length = rng.integers(0, rep_max_len)
        rep_start_pos = rng.integers(0, len(s) - rep_length + 1)
        s[rep_start_pos:rep_start_pos + rep_length] = [unknown_character] * rep_length
    return s



def build_dataset(config):

    if 'synthetic' in config['apply_pipelines']:
        print('### Building synthetic data')
        cm_path = config['rfam_cm_path']
        outpath = config['working_dir'] + '/' + 'synthetic_samples.fa'
        print('### Start sampling', config['n_samples_per_family'], 'sequences per cm-model from', cm_path)
        train = sample_synthetic_sequences(cm_path=cm_path,
                                           outpath=outpath,
                                           seed=config['seed'],
                                           n=config['n_samples_per_family'],
                                          )

        print('### Initial Synthetic training data frame has size', train.shape[0])

        if config['use_families']:
            print('### Select desired Families')
            train = train[train['family'].isin(config['use_families'])]
        if 'fold' in config['apply_pipelines']:
            print('### Apply Folding')
            train[['structure', 'energy']] = train.apply(lambda x: RNA.fold(''.join(x.sequence)), axis=1, result_type='expand')
            train.loc[:, 'origin'] = 'train'
            train.loc[:, 'pairs'] = train['structure'].apply(db2pairs)
            train.loc[:, 'pos1id'] = train['pairs'].apply(pairs2pos1)
            train.loc[:, 'pos2id'] = train['pairs'].apply(pairs2pos2)
            train.loc[:, 'pk'] = train['pairs'].apply(pairs2pk)
            train.loc[:, 'gc_content'] = train['sequence'].apply(lambda x: (''.join(x).count('G') + ''.join(x).count('C')) / len(x))
            train.loc[:, 'has_nc'] = train.apply(has_nc, axis=1)
            train.loc[:, 'has_pk'] = False
            train.loc[:, 'has_multiplet'] = False
            train.loc[:, 'is_pdb'] = False
            train = train.reset_index(drop=True)
            train.loc[:, 'Id'] = train.index
            train.loc[:, 'length'] = train['sequence'].apply(len)
        else:
            train.loc[:, 'origin'] = 'test'
            train.loc[:, 'pairs'] = train['sequence'].apply(lambda x: [])
            train.loc[:, 'pos1id'] = train['sequence'].apply(lambda x: [])
            train.loc[:, 'pos2id'] = train['sequence'].apply(lambda x: [])
            train.loc[:, 'pk'] = train['sequence'].apply(lambda x: [])
            train.loc[:, 'gc_content'] = train['sequence'].apply(lambda x: (''.join(x).count('G') + ''.join(x).count('C')) / len(x))
            train.loc[:, 'has_nc'] = False
            train.loc[:, 'has_pk'] = False
            train.loc[:, 'has_multiplet'] = False
            train.loc[:, 'is_pdb'] = False
            train = train.reset_index(drop=True)
            train.loc[:, 'Id'] = train.index
            train.loc[:, 'length'] = train['sequence'].apply(len)


    if not 'synthetic' in config['apply_pipelines']:
        print('### Try read initial data')
        trains = []
        for p in config['initial_train']:
            d = pd.read_pickle(p)
            trains.append(d)
        train = pd.concat(trains)
    print('### Number of samples in train set:', train.shape[0])

    if config['sample_valid_size'] != 0:
        valid = train.sample(n=config['sample_valid_size'], random_state=config['seed'])
        valid.loc[:, 'origin'] = 'valid'

        train = train[~train['Id'].isin(valid['Id'].unique())]

    print('### Apply sample selection criteria')

    if config['remove_n_only']:
        print('### Remove sequences without any canonical nucleotides')
        train.loc[:, 'not_seq_nc_only'] = train.apply(not_seq_nc_only, axis=1)
        train = train[train['not_seq_nc_only']]
        print('### Number of samples in train set:', train.shape[0])

    if config['cluster_train_internal']:
        print('### Start internal clustering of train data')
        cdhit = CDHIT(working_dir=config['working_dir'],
                      df=train,
                      )

        clusters = cdhit.cluster()
        print('### Number of clusters:', len(clusters))
        refs = clusters[clusters['ref'] == 1]
        train = train[train['Id'].isin(refs['Id'])]
        print('### Number of samples in train set:', train.shape[0])

    if config['n_structures_per_sequence']:
        print('### Sample', config['n_structures_per_sequence'],
              'for samples with more than',
              config['n_structures_per_sequence'],
              'structures per sequence')

        print('### Number of samples in train set:', len(train))

        train.loc[:, 'sequence'] = train['sequence'].apply(lambda x: ''.join(x))
        train.loc[:, 'dup_key'] = train.apply(gen_dup_key, axis=1)
        train = train.reset_index(drop=True)
        train.loc[:, 'Id'] = train.index
        same_sequences = defaultdict(list)
        df_list = []
        for name, group in train.groupby('sequence'):
            for key, g in group.groupby('dup_key'):
                same_sequences[name].append((key, len(g)))

        # print('### number of PDB samples in train set:', len(train[train['is_pdb']]))
        # print('### number of Multiplet samples in train:', len(train[train['has_multiplet']]))
        print('### Number of different sequences in train set:', len(same_sequences.keys()))
        print('### Count Number of different structures per sequence', Counter([len(v) for v in same_sequences.values()]))
        print('### Start sampling...')

        for name, group in train.groupby('sequence'):
            if len(same_sequences[name]) > n_structures_per_sequence:
                remaining_samples = 0
                # prioritize pdb
                if not group[group['is_pdb']].empty:
                    # prioritize multiplets
                    if not group[group['has_multiplet']].empty:
                        # enough multiplet samples -> sample from multiplets only
                        if len(group[group['has_multiplet']]) >= n_structures_per_sequence:
                            d = group[group['has_multiplet']].sample(n=n_structures_per_sequence, random_state=0)
                        # not enough multiplet samples -> sample as much as possible
                        else:
                            remaining_samples = n_structures_per_sequence - len(group[group['has_multiplet']])
                            d = group[group['has_multiplet']].copy()
                    else:
                        # enough pdb samples -> sample from pdb samples
                        if len(group[group['is_pdb']]) >= n_structures_per_sequence:
                            d = group[group['is_pdb']].sample(n=n_structures_per_sequence, random_state=0)
                        # not enough pdb samples -> sample as much as possible from pdb
                        else:
                            remaining_samples = n_structures_per_sequence - len(group[group['is_pdb']])
                            d = group[group['is_pdb']].copy()
                # no pdb samples -> sample from group directly
                else:
                    # prioritize pk samples
                    if not group[group['has_pk']].empty:
                        if len(group[group['has_pk']]) >= n_structures_per_sequence:
                            d = group[group['has_pk']].sample(n=n_structures_per_sequence, random_state=0)
                        else:
                            remaining_samples = n_structures_per_sequence - len(group[group['has_pk']])
                            d = group[group['has_pk']].copy()
                    else:
                        d = group.sample(n=n_structures_per_sequence, random_state=0)


                # if there were multiplet samples or pdb samples but not enough...
                if remaining_samples > 0:
                    # check if there are more pdb samples available without multiplets
                    if len(group[group['is_pdb']]) > len(group[group['has_multiplet']]):
                        pdb_no_multi = group[(group['is_pdb']) & ~(group['has_multiplet'])]
                        if len(pdb_no_multi) >= remaining_samples:
                            d2 = pdb_no_multi.sample(n=remaining_samples, random_state=0)
                            remaining_samples = 0
                        else:
                            remaining_samples = remaining_samples - len(pdb_no_multi)
                            d2 = pdb_no_multi.copy()
                            d2 = pd.concat([d2, group[~group['is_pdb']].sample(n=remaining_samples, random_state=0)])
                        if remaining_samples > 0:
                            no_pdb = group[~(group['is_pdb']) & ~(group['has_multiplet'])]
                            if not no_pdb[no_pdb['has_pk']].empty:
                                if len(no_pdb[no_pdb['has_pk']]) >= remaining_samples:
                                    d3 = no_pdb[no_pdb['has_pk']].sample(n=remaining_samples, random_state=0)
                                else:
                                    remaining_samples = remaining_samples - len(no_pdb[no_pdb['has_pk']])
                                    d3 = no_pdb[no_pdb['has_pk']].copy()
                            else:
                                d3 = no_pdb.sample(n=remaining_samples, random_state=0)
                            d2 = pd.concat([d2, d3])
                            if remaining_samples > 0:
                                no_pk = no_pdb[~no_pdb['has_pk']]
                                d3 = no_pk.sample(n=remaining_samples, random_state=0)
                                d2 = pd.concat([d2, d3])
                    else:
                        no_pdb = group[~(group['is_pdb']) & ~(group['has_multiplet'])]
                        d2 = no_pdb.sample(n=remaining_samples, random_state=0)

                    d = pd.concat([d, d2])

                df_list.append(d)
            else:
                df_list.append(group)
        train = pd.concat(df_list)
        train.loc[:, 'sequence'] = train['sequence'].apply(list)

        print('### Number of samples in train set after sampling:', len(train))

    print('### Number of samples in train set after selection:', train.shape[0])

    explicit_test = config['explicit_test']
    explicit_valid = config['explicit_valid']

    testsets = []
    for tset in explicit_test:
        t = pd.read_pickle(tset)
        testsets.append(t)
    try:
        test = pd.concat(testsets)
    except Exception as e:
        raise UserWarning('No test sets provided, stop data building.')

    validsets = []
    for vset in explicit_valid:
        v = pd.read_pickle(vset)
        validsets.append(v)
    if config['sample_valid_size'] != 0:
        validsets.append(valid)
    try:
        valid = pd.concat(validsets)
    except Exception as e:
        raise UserWarning('No valid set provided, stop data building.')

    train = train[config['initial_columns']]
    valid = valid[config['initial_columns']]
    test = test[config['initial_columns']]

    print('### Train has shape', train.shape)
    print('### Valid has shape', valid.shape)
    print('### Test has shape', test.shape)

    df = pd.concat([train, valid, test])

    df['is_pdb'] = df['is_pdb'].astype(bool)
    df['has_pk'] = df['has_pk'].astype(bool)
    df['has_nc'] = df['has_nc'].astype(bool)
    df['has_multiplet'] = df['has_multiplet'].astype(bool)

    print('### Total number of samples Data:', len(df))
    print('### Apply sample selection criteria')
    if config['pdb_only']:
        print('### Use only PDB samples')
        df = df[df['is_pdb']]
        print('### Number of samples in df set:', df.shape[0])

    if config['no_pdb']:
        print('### Remove PDB samples')
        df = df[~df['is_pdb']]
        print('### Number of samples in df set:', df.shape[0])

    if config['no_multiplets']:
        print('### Use only samples without multiplets')
        print(df['has_multiplet'])
        df = df[~df['has_multiplet']]
        print('### Number of samples in df set:', df.shape[0])

    if config['multiplets_only']:
        print('### remove samples with multiplets')
        df = df[df['has_multiplet']]
        print('### Number of samples in df set:', df.shape[0])

    if config['no_pk']:
        print('### Remove samples with pseudoknots')
        df = df[~df['has_pk']]
        print('### Number of samples in df set:', df.shape[0])

    if config['pk_only']:
        print('### Use only samples with pseudoknots')
        df = df[df['has_pk']]
        print('### Number of samples in df set:', df.shape[0])

    if config['no_nc']:
        print('### Remove samples with non-canonical pairs')
        df = df[~df['has_nc']]
        print('### Number of samples in df set:', df.shape[0])

    if config['nc_only']:
        print('### Use only samples that contain non-canonical pairs')
        df = df[df['has_nc']]
        print('### Number of samples in df set:', df.shape[0])

    df = df.reset_index(drop=True)
    df.loc[:, 'Id'] = df.index

    if config['resolve_iupac']:
        print('### Resolve IUPAC nucleotides')
        df = resolve_iupac(df)
        s = []
        for i, row in df.iterrows():
            s += row['sequence']
        print('### Count nucleotides after resolving IUPAC:', Counter(s))


    print(valid['origin'].unique())
    test = df[df['origin'].isin(test['origin'].unique())]
    valid = df[df['origin'].isin(valid['origin'].unique())]

    print(len(test))
    print(len(valid))

    if test.empty:
        raise UserWarning('Criteria result in no test data')

    if valid.empty:
        raise UserWarning('Criteria result in no valid data')


    train = df[~df['origin'].isin(test['origin'].unique())]
    print(len(train))

    train = train[~train['origin'].isin(valid['origin'].unique())]
    print(len(train))
    print('### Number of samples in Train set', train.shape)

    print('### Dataset Information:')
    df = pd.concat([train, valid, test])
    print(df.groupby(['origin']).count())

    pipeline_data = []
    pipeline_data.append(('initial_train', train.copy()))
    pipeline_data.append(('initial_valid', valid.copy()))
    pipeline_data.append(('initial_test', test.copy()))

    if 'design' in config['apply_pipelines']:
        print('### Apply design pipeline')
        print('### Number of samples in train set:', len(train))
        print('### Number of samples in valid set:', len(valid))
        print('### Number of samples in test set:', len(test))
        test.loc[:, 'key'] = test.apply(lambda x: '_'.join([str(i[0]) for i in x.pairs] + [str(i[1]) for i in x.pairs]), axis=1)
        valid.loc[:, 'key'] = valid.apply(lambda x: '_'.join([str(i[0]) for i in x.pairs] + [str(i[1]) for i in x.pairs]), axis=1)
        train.loc[:, 'key'] = train.apply(lambda x: '_'.join([str(i[0]) for i in x.pairs] + [str(i[1]) for i in x.pairs]), axis=1)
        # valid.loc[:, 'key'] = valid.apply(lambda x: '_'.join([str(i) for i in x.pos1id] + [str(i) for i in x.pos2id]), axis=1)
        # train.loc[:, 'key'] = train.apply(lambda x: '_'.join([str(i) for i in x.pos1id] + [str(i) for i in x.pos2id]), axis=1)
        print('### Remove samples from valid that are in test')
        test_keys = list(test['key'].unique())

        valid = valid[~valid['key'].isin(test_keys)]
        print('### Number of samples in valid set:', len(valid))

        print('### Remove samples from train with the same structure in test or valid')
        test_valid_keys = test_keys + list(valid['key'].unique())

        train = train[~train['key'].isin(test_valid_keys)]

        print('### Number of samples in train:', len(train))

        rng = np.random.default_rng(seed=config['seed'])

        valid.loc[:, 'sequence'] = valid['sequence'].apply(lambda x: get_task_seq(config['design_task'], x, rng))
        train.loc[:, 'sequence'] = train['sequence'].apply(lambda x: get_task_seq(config['design_task'], x, rng))


    if 'ss80' in config['apply_pipelines']:

        print('### Apply 80% sequence similarity pipeline between sets')
        train_valid = pd.concat([train, valid])
        print('### Number of samples in train+valid: ', train_valid.shape[0])

        # remove sequence similarity to test
        pipeline = SequenceSimilarityPipeline(working_dir=config['working_dir'],
                                                       sets2keep=test,
                                                       sets2reduce=train_valid,
                                                      )
        pipeline.apply_sequence_similarity()

        train_valid = pipeline.reduce
        print('### Number of samples in train+valid after removing similarity: ', train_valid.shape[0])

        valid = train_valid[~train_valid['origin'].str.contains('train')]
        train = train_valid[train_valid['origin'].str.contains('train')]

        ss80_1_train_data = ('ss80_1_train', train.copy())
        ss80_1_valid_data = ('ss80_1_valid', valid.copy())

        pipeline_data.append(ss80_1_train_data)
        pipeline_data.append(ss80_1_valid_data)

        print('### Number of samples in valid', valid.shape[0])
        print('### Number of samples in train', train.shape[0])

        print('### Apply sequence similarity between train and test + valid')
        test_valid = pd.concat([test, valid])
        print('### Number of samples in test + valid', len(test_valid))

        pipeline = SequenceSimilarityPipeline(working_dir=config['working_dir'],
                                                       sets2keep=test_valid,
                                                       sets2reduce=train,
                                                      )
        pipeline.apply_sequence_similarity()

        train = pipeline.reduce

        ss80_2_train_data = ('ss80_2_train', train.copy())
        ss80_2_valid_data = ('ss80_2_valid', valid.copy())

        pipeline_data.append(ss80_2_train_data)
        pipeline_data.append(ss80_2_valid_data)

        s = []
        for i, row in train.iterrows():
            s += row['sequence']
        print('### Count nucleotides in train after ss80:', Counter(s))

        s = []
        for i, row in valid.iterrows():
            s += row['sequence']
        print('### Count nucleotides in valid after ss80:', Counter(s))

        s = []
        for i, row in test.iterrows():
            s += row['sequence']
        print('### Count nucleotides in test after ss80:', Counter(s))

        print('### Number of samples in train after sequence similarity pipeline', train.shape[0])

    if 'blast' in config['apply_pipelines']:
        print('### Start BLAST search of test samples against train and valid')
        train_valid = pd.concat([train, valid])
        vsets = '_'.join(valid['origin'].unique())
        e_value = 10
        hits = []
        
        per_sample_train_valid = train_valid.copy()

        # build database of train_valid to query with test
        fasta_path = Path(f"{config['working_dir']}/train_{vsets}.fasta")
        df2fasta(train_valid, fasta_path)

        subprocess.call(["./makeblastdb", "-in", str(fasta_path.resolve()), "-dbtype", "nucl"], cwd=config['ncbi_bin_dir'])

        for dset, data in test.groupby('origin'):
            if 'spot1' in config['apply_pipelines']:
                if 'bprna_ts0' in dset:
                    print('### Skip TS0')
                    continue
            for i, sample in data.iterrows():
                test_sample_path = Path(f"{config['working_dir']}/{dset}_{i}.fasta")
                with open(test_sample_path, 'w+') as f:
                    f.write(f">{i}\n")
                    f.write(''.join(sample['sequence']) + '\n')
                tbl_path = Path(Path(f"{config['working_dir']}/{dset}_{i}.tbl"))
                query = ''.join(sample['sequence'])

                subprocess.call(['./blastn', '-db', str(fasta_path.resolve()), '-query', str(test_sample_path.resolve()), '-out', str(tbl_path.resolve()), '-evalue', str(e_value), '-outfmt', "7 qacc sacc pident nident score evalue bitscore qstart qend sstart send qseq sseq "], cwd=config['ncbi_bin_dir'])

                tab_parser = BlastTabParser(tbl_path, query=query, comments=True)
                tv_hits = tab_parser.parse()
                if tv_hits is not None:
                    print('### Hits with blast search for dataset:', dset, 'Id:', i, ':', len(tv_hits))
                    hits.append(tv_hits)
                    #TODO: print if the sequence is in ts1 or ts3 and also in ts_hard: requires that we know the sequences of ts_hard.
                else:
                    print('### No hits found for dataset', dset, 'Id', i)
                
                if tv_hits is not None:
                    current_sample_hit_list = [int(x.split('-')[0]) for x in tv_hits['Id'].unique()]
                    per_sample_train_valid = per_sample_train_valid[~per_sample_train_valid['Id'].isin(current_sample_hit_list)]
                    per_sample_train = per_sample_train_valid[per_sample_train_valid['origin'].str.contains('train')]
                    per_sample_valid = per_sample_train_valid[~per_sample_train_valid['origin'].str.contains('train')]
                    pipeline_data.append((f'blast_{dset}_{i}_train', per_sample_train.copy()))
                    pipeline_data.append((f'blast_{dset}_{i}_valid', per_sample_valid.copy()))
                else:
                    per_sample_train = per_sample_train_valid[per_sample_train_valid['origin'].str.contains('train')]
                    per_sample_valid = per_sample_train_valid[~per_sample_train_valid['origin'].str.contains('train')]
                    pipeline_data.append((f'blast_{dset}_{i}_train', per_sample_train.copy()))
                    pipeline_data.append((f'blast_{dset}_{i}_valid', per_sample_valid.copy()))


            # get intermediate train and valid data
            intermediate_hits = pd.concat(hits)
            intermediate_hit_list = list(set([int(x.split('-')[0]) for x in intermediate_hits['Id'].unique()]))
            intermediate_train_valid = train_valid[~train_valid['Id'].isin(intermediate_hit_list)]
            intermediate_train = intermediate_train_valid[intermediate_train_valid['origin'].str.contains('train')]
            intermediate_valid = intermediate_train_valid[~intermediate_train_valid['origin'].str.contains('train')]

            pipeline_data.append((f'blast_{dset}_train', intermediate_train.copy()))
            pipeline_data.append((f'blast_{dset}_valid', intermediate_valid.copy()))

        if hits:
            hits = pd.concat(hits)
            hit_list = list(set([int(x.split('-')[0]) for x in hits['Id'].unique()]))
            print('### Total number of hits in blast search', len(hit_list))
    
            print('### Number of samples in train+valid before removing hits:', len(train_valid))
            train_valid = train_valid[~train_valid['Id'].isin(hit_list)]
            print('### Number of samples in train+valid:', len(train_valid))
    
            train = train_valid[train_valid['origin'].str.contains('train')]
            print('### Train has size', len(train))
            valid = train_valid[~train_valid['origin'].str.contains('train')]
            print('### Valid has size', len(valid))
    
            s = []
            for i, row in train.iterrows():
                s += row['sequence']
            print('### Count nucleotides in train after blast:', Counter(s))
    
            s = []
            for i, row in valid.iterrows():
                s += row['sequence']
            print('### Count nucleotides in valid after blast:', Counter(s))
    
            s = []
            for i, row in test.iterrows():
                s += row['sequence']
            print('### Count nucleotides in test after blast:', Counter(s))
        else:
            train = train_valid[train_valid['origin'].str.contains('train')]
            print('### Train has size', len(train))
            valid = train_valid[~train_valid['origin'].str.contains('train')]
            print('### Valid has size', len(valid))
    
            s = []
            for i, row in train.iterrows():
                s += row['sequence']
            print('### Count nucleotides in train after blast:', Counter(s))
    
            s = []
            for i, row in valid.iterrows():
                s += row['sequence']
            print('### Count nucleotides in valid after blast:', Counter(s))
    
            s = []
            for i, row in test.iterrows():
                s += row['sequence']
            print('### Count nucleotides in test after blast:', Counter(s))




    # 6. sss
    if 'sss' in config['apply_pipelines']:
        print('### Apply sequence and structure similarity pipeline')

        train_valid = pd.concat([train, valid])

        train_intermediate = train.copy()
        valid_intermediate = valid.copy()

        train_intermediate.loc[:, 'Id'] = train_intermediate['Id'].apply(str)
        valid_intermediate.loc[:, 'Id'] = valid_intermediate['Id'].apply(str)


        print('### Number of samples in train_valid', len(train_valid))

        fasta_path = f"{config['working_dir']}/train_valid.fasta"
        df2fasta(train_valid, fasta_path)
        infernal = Infernal(working_dir=config['working_dir'],
                            E=0.1,
                            incE=0.1)

        hit_list = []
        cms = []
        for p in config['cms_from']:
            cms += list(Path(p).glob('*.cm'))
        # 'ts1_ts2_ts3_ts_hard_puzzles24.cm'
        for i, cm in enumerate(cms, 1):
            try:
                # hit_list += infernal.search_database(cm_database=config['cm_database'], identifier=str(cm.stem), fasta_db=fasta_path)
                # hit_list += infernal.search_database(cm_database=cm, identifier=str(cm.stem), fasta_db=fasta_path)
                current_hits = infernal.search_database(cm_database=cm, identifier=str(cm.stem), fasta_db=fasta_path)
                print('### Found', len(current_hits), 'hits for CM', cm.stem, 'from', cm.parents[0])
                hit_list += current_hits
                #TODO: If cm is ts_hard, check where the sequence originates from, ts1, or ts3... and print this.

                # train_intermediate = train_valid[train_valid['origin'].str.contains('train')]
                # valid_intermediate = train_valid[~train_valid['origin'].str.contains('train')]

                train_intermediate = train_intermediate[~train_intermediate['Id'].isin(current_hits)]
                valid_intermediate = valid_intermediate[~valid_intermediate['Id'].isin(current_hits)]

                pipeline_data.append((f"cm_{str(cm.parents[0]).split('/')[-1]}_{cm.stem}_train", train_intermediate.copy()))
                pipeline_data.append((f"cm_{str(cm.parents[0]).split('/')[-1]}_{cm.stem}_valid", valid_intermediate.copy()))
                


            except ValueError as e:
                print(e)
            except Exception as e:
                print(e)
                continue
            hit_list = list(set(hit_list))
            print('### Number of non-redundant hits so far:', len(hit_list))

            print('### Found', len(hit_list), f'hits after searching {i}/{len(cms)} CMs.')
            
        print('### Total number of hits', len(hit_list))

        train_valid.loc[:, 'Id'] = train_valid['Id'].apply(str)
        if hit_list:
            train_valid.loc[:, 'sss'] = ~train_valid['Id'].isin(hit_list)
        print('### Number of samples in train+valid after sequence and similarity pipeline:', train_valid[train_valid['sss']].shape)

        test.loc[:, 'sss'] = True

        train_valid = train_valid[train_valid['sss']]

        train = train_valid[train_valid['origin'].str.contains('train')]
        print('### Train has size', len(train))
        valid = train_valid[~train_valid['origin'].str.contains('train')]
        print('### Valid has size', len(valid))

        pipeline_data.append(('sss_train', train.copy()))
        pipeline_data.append(('sss_valid', valid.copy()))

        s = []
        for i, row in train.iterrows():
            s += row['sequence']
        print('### Count nucleotides in train after sss:', Counter(s))

        s = []
        for i, row in valid.iterrows():
            s += row['sequence']
        print('### Count nucleotides in valid after sss:', Counter(s))

        s = []
        for i, row in test.iterrows():
            s += row['sequence']
        print('### Count nucleotides in test after sss:', Counter(s))



    print('### Dumping datasets')
    if not test.empty:
        test = test[config['final_columns']]
        test = test.reset_index(drop=True)
        test.loc[:, 'Id'] = test.index
        with open(f"{config['out_dir']}/{config['dataset_name']}_benchmark.pkl", 'wb') as f:
            pickle.dump(test, f)
    if not valid.empty:
        valid = valid[config['final_columns']]
        valid = valid.reset_index(drop=True)
        valid.loc[:, 'Id'] = valid.index
        with open(f"{config['out_dir']}/{config['dataset_name']}_valid.pkl", 'wb') as f:
            pickle.dump(valid, f)


    train = train[config['final_columns']]

    train = train.reset_index(drop=True)
    train.loc[:, 'Id'] = train.index
    print(train)

    with open(f"{config['out_dir']}/{config['dataset_name']}_train.pkl", 'wb') as f:
        pickle.dump(train, f)

    print('### Final Number of samples in train set:', train.shape[0])
    print('### Final Number of samples in valid set:', valid.shape[0])
    print('### Final Number of samples in test set:', test.shape[0])

    print('########################################################################')
    print('### PIPELINE STATS ###')
    print()
    print('name', 'num_samples', 'PDB', 'non-PDB')
    print()
    for name, data in pipeline_data:
        d_pdb = data[data['is_pdb']]
        n_pdb = data[~data['is_pdb']]
        print(name, data.shape[0], d_pdb.shape[0], n_pdb.shape[0])
    print()
    print('########################################################################')

    import torch

    torch.save(pipeline_data, f"{config['out_dir']}/{config['dataset_name']}_pipeline_stats.pt")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='data_configs/dataset_config.yml', type=str, help='The Path to the config for the dataset')

    args = parser.parse_args()

    import yaml

    with open(args.config_path, 'r') as f:
        fi = f.read()
        # fi = fi.replace('!!', '#') # irgnore yaml tags
        config = yaml.load(fi, Loader=yaml.Loader)

    print('### Build dataset for config', args.config_path)

    print('### Config:')

    print()
    print(config)
    print()

    build_dataset(config)