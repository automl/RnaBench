import json

import pandas as pd
import RnaBench

from typing import Union, List, Optional
from pathlib import Path
from Bio import SeqIO, PDB

from RnaBench.lib.data.similarity import Infernal, LocARNA, CDHIT2D, CDHIT, RNAAlifold, MMSEQS2
from RnaBench.lib.utils import df2fasta
from RnaBench.lib.data.parser.infernal_tblout_parser import InfernalTbloutParser
from RnaBench.lib.data.parser.pdb_parser import MMCIF3DParser, MMCIF2DParser
from RnaBench.lib.data.structure_annotation import DSSR, BpRNA


class SequenceStructureSimilarityPipeline():
    """
    Pipeline for data preprocessing to remove similarity between different splits.

    TODO
    """
    def __init__(self,
                 working_dir : Union[Path, str],
                 sets2keep : List[pd.DataFrame],
                 sets2reduce : List[pd.DataFrame],
                 ):
        """
        TODO
        """
        self._working_dir : Union[Path, str] = working_dir

        if isinstance(sets2keep, list):
            self._keep : pd.DataFrame = pd.concat(sets2keep)
        else:
            self._keep : pd.DataFrame = sets2keep

        if isinstance(sets2reduce, list):
            self._reduce : pd.DataFrame = pd.concat(sets2reduce)
        else:
            self._reduce : pd.DataFrame = sets2reduce

        self._similarity_method : RnaBench.core.similarity.CDHIT2D = CDHIT2D(working_dir = working_dir,
                                                                             df_keep = self._keep,
                                                                             df_reduce = self._reduce,
                                                                             )

        self._alignment_method : RnaBench.core.similarity.LocARNA = LocARNA(working_dir = working_dir)

        # TODO: Add other parameters
        self._profile_method : RnaBench.core.similarity.Infernal = Infernal(working_dir = working_dir)


        self._clustering_method : RnaBench.core.similarity.MMSEQS2 = MMSEQS2(working_dir = working_dir)

        self._folding_method : RnaBench.core.similarity.RNAAlifold = RNAAlifold()



    def apply_sequence_similarity(self):
        """
        Remove sequences from data based on sequence similarity only.
        """
        id_type = type(self._reduce.iloc[0]['Id'])
        self._reduce.loc[:, 'Id'] = self._reduce['Id'].apply(str)
        self._similarity_method.dfs2fasta()
        self._similarity_method.cluster()
        self._reduce = self._reduce[self._reduce['Id'].isin(self._similarity_method.non_similar_ids())]
        self._reduce.loc[:, 'Id'] = self._reduce['Id'].apply(id_type)

    def cluster(self,
                df : pd.DataFrame,
                ) -> pd.DataFrame:
        """
        get_clusters from a given dataframe.
        """
        clustered_df = self._clustering_method.easy_cluster_df(df=df,
                                              fasta_file='2b_clustered.fa',
                                              output_file_stem='2b_clustered'
                                              )
        assert not any(clustered_df['cluster'].isna())

        print('### Number of clusters in dataset:', len(clustered_df['cluster'].unique()))

        return clustered_df

    def generate_covariance_models(self,
                                   clustered_df: pd.DataFrame,
                                   dataset : str,
                                   outdir : str,
                                   split : str = 'test',
                                   ) -> List[int]:
        """
        Generate covaraince models from a given clustered dataframe.
        """

        cm_dir = f'{outdir}/{dataset}_{split}'

        Path(cm_dir).mkdir(exist_ok=True, parents=True)

        cluster_ids = []

        for name, d in clustered_df.groupby('cluster'):
            print('### Size of cluster: ', d.shape[0])
            cluster_ids.append(name)
            if not Path(f"{cm_dir}/{name}.cm").is_file():
                df2fasta(df=d, out_path=f"{self._working_dir}/{name}.fa")
                if d.shape[0] == 1:
                    SeqIO.convert(f"{self._working_dir}/{name}.fa", "fasta", f"{self._working_dir}/{name}.stk", "stockholm")
                    stk_path = f"{self._working_dir}/{name}.stk"
                else:
                    self._alignment_method.align_multi(fasta_file=f"{name}.fa")
                    stk_path = f"{self._working_dir}/{name}.out/results/result.stk"
                # subprocess.call(["RNAalifold", stk_path, f"--aln-stk={name}_ali"])
                self._folding_method.fold(stk_path = stk_path, aln_stk_prefix = f"{name}_ali")
                self._profile_method.build(stk_path=f"{name}_ali.stk", out_stem=name, name=f"{name}_{dataset}_{split}", cm_dir=cm_dir)
                Path(f"{name}_ali.stk").unlink()
            else:
                print('### CM for cluster', name, 'already exists. No need to build new one.')

        return cluster_ids

    def calibrate_covariance_model(self,
                                   cluster_ids,
                                   outdir : str,
                                   dataset : str,
                                   split : str = 'test',
                                   ):
        """
        Calibrate a given covariance model.
        """
        for i in cluster_ids:
            self._profile_method.calibrate(cm_path=f"{outdir}/{dataset}_{split}/{i}.cm")


    def generate_hmms_no_sec_struc(self,
                                   clustered_df: pd.DataFrame,
                                   dataset : str,
                                   outdir : str,
                                   split : str = 'test',
                                   ) -> List[int]:
        """
        Generate covaraince models from a given clustered dataframe.
        """

        cm_dir = f'{outdir}/{dataset}_{split}'

        Path(cm_dir).mkdir(exist_ok=True, parents=True)

        cluster_ids = []

        for name, d in clustered_df.groupby('cluster'):
            print('### Size of cluster: ', d.shape[0])
            cluster_ids.append(name)
            if not Path(f"{cm_dir}/{name}.cm").is_file():
                df2fasta(df=d, out_path=f"{self._working_dir}/{name}.fa")
                if d.shape[0] == 1:
                    SeqIO.convert(f"{self._working_dir}/{name}.fa", "fasta", f"{self._working_dir}/{name}.stk", "stockholm")
                    stk_path = f"{self._working_dir}/{name}.stk"
                else:
                    self._alignment_method.align_multi(fasta_file=f"{name}.fa")
                    stk_path = f"{self._working_dir}/{name}.out/results/result.stk"
                self._profile_method.build_noss(stk_path=stk_path, out_stem=name, name=f"{name}_{dataset}_{split}", cm_dir=cm_dir)
            else:
                print('### CM for cluster', name, 'already exists. No need to build new one.')

        return cluster_ids


    def apply_sequence_and_structure_similarity(self,
                                                cm_path : Union[Path, str],
                                                cm_id : int,
                                                fasta_path : Union[str, Path],
                                                ):
        """
        search sequences against a CM to find hits that have to be removed.
        """
        outfile = f"{self._working_dir}/{cm_id}.aln"
        hit_list = self._profile_method.search(cm_path=cm_path, fasta_path=fasta_path, outfile=outfile)
        return hit_list

    @property
    def reduce(self):
        return self._reduce


class SequenceSimilarityPipeline():
    """
    Pipeline for data preprocessing to remove similarity between different splits.

    TODO
    """
    def __init__(self,
                 working_dir : Union[Path, str],
                 sets2keep : List[pd.DataFrame],
                 sets2reduce : List[pd.DataFrame],
                 ):
        """
        TODO
        """
        self._working_dir : Union[Path, str] = working_dir

        if isinstance(sets2keep, list):
            self._keep : pd.DataFrame = pd.concat(sets2keep)
        else:
            self._keep : pd.DataFrame = sets2keep

        if isinstance(sets2reduce, list):
            self._reduce : pd.DataFrame = pd.concat(sets2reduce)
        else:
            self._reduce : pd.DataFrame = sets2reduce

        self._similarity_method : RnaBench.core.similarity.CDHIT2D = CDHIT2D(working_dir = working_dir,
                                                                             df_keep = self._keep,
                                                                             df_reduce = self._reduce,
                                                                             )


    def apply_sequence_similarity(self):
        """
        Remove sequences from data based on sequence similarity only.
        """
        id_type = type(self._reduce.iloc[0]['Id'])
        self._reduce.loc[:, 'Id'] = self._reduce['Id'].apply(str)
        self._similarity_method.dfs2fasta()
        self._similarity_method.cluster()
        self._reduce = self._reduce[self._reduce['Id'].isin(self._similarity_method.non_similar_ids())]
        self._reduce.loc[:, 'Id'] = self._reduce['Id'].apply(id_type)

    @property
    def reduce(self):
        return self._reduce




class FamilyBasedSimilarityPipeline():
    """
    The class implements a pipeline that assigns RNA families given in the
    Rfam database to all sequences of the provided data.
    It uses pre-calibrated CMs from Rfam to search againstz the given sequence
    database.

    Additionally allows to removes sequences from one provided set with respect to families
    in another set.
    """
    def __init__(self,
                 query_fasta_path : str,
                 cm_path : str,
                 clanin_path: str,
                 tbl_outpath : str,
                 working_dir : str = 'working_dir',
                 ):
        self._query_path : str = query_fasta_path
        self._cm_path : str = cm_path
        self._clanin_path : str = clanin_path
        self._tbl_outpath : str = tbl_outpath

        self._infernal : RnaBench.core.similarity.Infernal = Infernal(working_dir=working_dir)
        self._parser : RnaBench.core.parser.infernal_tblout_parser.InfernalTbloutParser = InfernalTbloutParser(self._tbl_outpath)

    def get_rfam_families(self):
        self._infernal.get_family_information(
          queries_fasta_path = self._query_path,
          cm_path = self._cm_path,
          clanin_path = self._clanin_path,
          outpath = self._tbl_outpath,
        )

        family_info = self._parser.parse()

        return family_info

class PDB2DPipeline():
    """
    Obtain 2D structure information from pdb mmcif files.
    """
    def __init__(self,
                 mmcif_dir : str,
                 dssr_dir : str,
                 bprna_dir : str,
                 working_dir : str,
                 file_type : str = 'mmcif',
                ):
        self._mmcif_dir : str = mmcif_dir
        self._dssr_dir : str = dssr_dir
        self._bprna_dir : str = bprna_dir
        self._working_dir : str = working_dir
        self._file_type : str = file_type

        self._parser : RnaBench.core.parser.MMCIF2DParser = MMCIF2DParser(file_type=file_type)
        self._dssr : RnaBench.core.structure_annotation.DSSR = DSSR(dssr_dir)
        self._bprna : RnaBench.core.structure_annotation.BpRNA = BpRNA(bprna_dir,
                                                                       working_dir,
                                                                      )

    def process_mmcif_file(self, mmcif_path):
        id_features = {}

        self._parser.read_structure(mmcif_path)

        mmcif_header = self._parser.get_header()
        id_features['header'] = mmcif_header
        self._parser.get_rna_chain_ids()
        id_features['RNA_chains'] = self._parser.chains

        self._dssr.run(mmcif_path)
        nuc_pair_dict = self._dssr.parse_json()
        id_features.update(nuc_pair_dict)

        return id_features

    def process_mmcif_dir(self):
        # print('### Start PDB Data processing for MMCIF directory', mmcif_dir)
        if self._file_type == 'mmcif':
            mmcif_files = list(Path(self._mmcif_dir).glob('*.cif'))
        else:
            mmcif_files = list(Path(self._mmcif_dir).glob('*.pdb'))
        # print(list(mmcif_files))
        # structure_parser = MMCIF2DParser()
        # dssr = DSSR(dssr_dir)
        ct_path = Path(self._dssr_dir, 'dssr-2ndstrs.ct')
        structure_information = {}
        for i, p in enumerate(mmcif_files, 1):
            # parse also with biopython to get the sequence correct.
            # Also correct the pairs afterwards to fit the length of the sequence

            print(i)
            print('### Process', p.stem)
            print('### Sequence:', self._parser.get_sequence(p))
            if Path(self._working_dir, f"{p.stem}.json").is_file():
                print('### Read from existing JSON')
                with open(Path(self._working_dir, f"{p.stem}.json"), 'r') as f:
                    id_features = json.load(f)
            else:
                try:
                    id_features = self.process_mmcif_file(p)
                    # print(id_features)
                    # print('### Writing features to JSON')
                    with open(Path(self._working_dir, f"{p.stem}.json"), 'w') as f:
                        json.dump(id_features, f)
                except Exception as e:
                    print(e)
                    continue

            structure_information[p.stem] = id_features
            # print('### Done.')

        # print('### Start Building Dataframe')
        dfs = []

        for pdb_id, infos in structure_information.items():
            id_dicts = []
            if not infos['pairs']:
                continue
            # print('### Build DataFrame for', pdb_id)
            single_chains = [c for c in infos['sequences'].keys() if not c == 'all']
            rna_chains = infos['RNA_chains']
            can_use_all = all(c in rna_chains for c in single_chains)
            if can_use_all:
                for chain, pairs in infos['pairs'].items():
                    pos1id = [p[0] for p in pairs]
                    pos2id = [p[1] for p in pairs]
                    sequence = infos['sequences'][chain]
                    release_date = infos['header']['deposition_date']
                    # msa_id = infos['header']['idcode'] + '-' + chain
                    msa_id = str(pdb_id) + '-' + chain
                    method = infos['header']['structure_method']
                    resolution = infos['header']['resolution']
                    name = infos['header']['name']
                    head = infos['header']['head']
                    dataset = 'PDBall_rna'
                    split = 'train'

                    id_dicts.append({
                        'pos1id': pos1id,
                        'pos2id': pos2id,
                        'sequence': list(''.join(sequence).upper()),
                        'release_date': release_date,
                        'msa_id': msa_id,
                        'method': method,
                        'resolution': resolution,
                        'name': name,
                        'head': head,
                        'dataset': dataset,
                        'set': split,
                        'is_pdb': True,
                        'missing_chains': False,
                    })
            else:
                for chain, pairs in infos['pairs'].items():
                    if chain in rna_chains:
                        pos1id = [p[0] for p in pairs]
                        pos2id = [p[1] for p in pairs]
                        sequence = infos['sequences'][chain]
                        release_date = infos['header']['deposition_date']
                        # msa_id = infos['header']['idcode'] + '-' + chain
                        msa_id = str(pdb_id) + '-' + chain
                        method = infos['header']['structure_method']
                        resolution = infos['header']['resolution']
                        name = infos['header']['name']
                        head = infos['header']['head']
                        dataset = 'PDBall_rna'
                        split = 'train'

                        id_dicts.append({
                            'pos1id': pos1id,
                            'pos2id': pos2id,
                            'sequence': list(''.join(sequence).upper()),
                            'release_date': release_date,
                            'msa_id': msa_id,
                            'method': method,
                            'resolution': resolution,
                            'name': name,
                            'head': head,
                            'dataset': dataset,
                            'set': split,
                            'is_pdb': True,
                            'missing_chains': True,
                        })

            df = pd.DataFrame(id_dicts)
            dfs.append(df)

        final_df = pd.concat(dfs)
        print(final_df)


        return final_df

        # with open(Path(out_dir, 'pdb_all.plk'), 'wb') as f:
        #     pickle.dump(final_df, f)

    def get_sequence(self):
        return self._parser.get_sequence()




if __name__ == '__main__':
    # dataset = 'e2efold_rnastralign'
    # dataset = 'e2efold_archiveii'
    # dataset = 'icml_v3'  # no 'is_pdb' column
    # dataset = 'riboformer_data_new_msa'  # only ~45000 samples
    # dataset = 'riboformer_data_new' # ~110k samples; very promising frame; structures with level descriptions only
    # dataset = 'riboformer_data_new_structure_m'  # ~82k samples; structures with BP, M, ...
    dataset = 'riboformer_data'  # ~110k samples; structures with BP, M, ...
    working_dir = Path('working_dir')
    # df = pd.read_pickle(f'data/{dataset}.plk.gz')
    # df = pd.read_pickle('/home/fred/github/rna_design/rna_data/dataframes/all_data.plk')
    df = pd.read_pickle(f'data/{dataset}.plk')
    print(df)
    print(df.columns)
    df_keep = df[(df['set'] == 'test') & (df['is_pdb'])]
    print(df_keep)
    print(df['dataset'])

    # df_reduce = df[df['set'] == 'train']

    # pipeline = SequenceStructureSimilarityPipeline(working_dir, df_keep, df_reduce)
    # pipeline.apply_sequence_similarity()

    # clusters = pipeline.cluster(df_keep)

    #cluster_ids = pipeline.generate_covariance_models(clustered_df = clusters,
    #                                                  dataset = dataset,
    #                                                  outdir = 'test_covariance',
    #                                                  split = 'test',
    #                                                 )

    # pipeline.calibrate_covariance_model(
    #                                     cluster_ids,
    #                                     outdir = 'test_covariance',
    #                                     dataset = dataset,
    #                                     split = 'test',
    #                                    )
