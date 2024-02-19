import subprocess
import pandas as pd

from pathlib import Path
from itertools import zip_longest
from typing import Union, Optional
from RnaBench.lib.data.parser.cd_hit_clstr_parser import CDHITClusterParser
from RnaBench.lib.utils import stockholm2idlist, df2fasta


def get_scores_and_refs(df):
    refs = []
    scores = []

    for score in df['sim']:
        if score == '*':
            refs.append(1)
            scores.append(100.0)
            continue
        refs.append(0)
        _, score = score.split('/')
        scores.append(float(score[:-1]))
    df['score'] = scores
    df['ref'] = refs

    return df


class CDHIT2D():
    def __init__(self, working_dir, df_keep, df_reduce, cluster_type='std', sim=0.8, n=5, l1=0.0, l2=0.0):
        self.df_keep = df_keep
        self.df_reduce = df_reduce

        Path(working_dir).mkdir(exist_ok=True, parents=True)

        self.fasta_path_keep = Path(working_dir, 'cluster_keep.fasta')
        self.fasta_path_reduce = Path(working_dir, 'cluster_reduce.fasta')
        self.out_path = Path(working_dir, 'cluster2D')

        self.std_args = ['-c', str(sim), '-T', '0', '-n', str(n), '-s', str(l1), '-s2', str(l2), '-g', '0', '-r', '0']

        self.cluster_type = cluster_type

        self.dfs2fasta()


    def dfs2fasta(self):
        with open(self.fasta_path_keep, 'w+') as f:
            for i, sequence in zip(self.df_keep['Id'], self.df_keep['sequence']):
                f.write('>'+str(i)+'\n')
                f.write(''.join(sequence).upper()+'\n')
        with open(self.fasta_path_reduce, 'w+') as f:
            for i, sequence in zip(self.df_reduce['Id'], self.df_reduce['sequence']):
                f.write('>'+str(i)+'\n')
                f.write(''.join(sequence).upper()+'\n')
            # for i, row in self.df_reduce.iterrows():
            #     f.write('>'+str(i)+'\n')
            #     f.write(''.join(row['sequence']).upper()+'\n')



    def cluster(self):
        base_args = ['cd-hit-est-2d',
                     '-i', str(self.fasta_path_keep.resolve()),
                     '-i2', str(self.fasta_path_reduce.resolve()),
                     '-o', str(self.out_path.resolve())]

        if self.cluster_type == 'std':
            args = base_args + self.std_args
        else:
            args = base_args + self.short_args

        p = subprocess.call(args)  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def non_similar_ids(self):
        keep = []
        with open(self.out_path, 'r') as f:
            for idx, seq in zip_longest(*[f]*2):
                idx = idx[1:].strip()
                seq = seq.strip()
                keep.append(idx)
        # train = train[train.index.isin(keep)]
        return keep



class CDHIT():
    def __init__(self,
                 working_dir,
                 df,
                 similarity=0.8,
                 word_size=5,  # -n 8,9,10 for thresholds 0.90 ~ 1.0 ; -n 7 for thresholds 0.88 ~ 0.9 ; -n 6 for thresholds 0.85 ~ 0.88 ; -n 5 for thresholds 0.80 ~ 0.85 ; -n 4 for thresholds 0.75 ~ 0.8
                 coverage=0.0,
                 length_cutoff=0.0):

        working_dir = Path(working_dir)
        working_dir.mkdir(exist_ok=True, parents=True)
        self.fasta_path = Path(working_dir, f"to_be_clustered.fa")
        self.ana_path = Path(working_dir, 'ana' + '.ana')
        self.ana_clstr_path = Path(working_dir, 'ana' + '.ana.clstr')
        self.csv_path = Path(working_dir, 'csv_out' + '.csv')
        self.df = df

        self.sim_ws = [(0.8, 5), (0.85, 6), (0.88, 7), (0.9, 8), (0.95, 9), (0.98, 10), (0.99, 10)]

        df2fasta(df[['Id', 'sequence']], self.fasta_path)

        self.similarity = similarity
        self.word_size = word_size
        self.coverage = coverage
        self.length_cutoff = length_cutoff

        self.working_dir = working_dir


    def df2fasta(self, df):
        with open(self.fasta_path, 'w+') as f:
            for _, row in df.iterrows():
                f.write('>' + str(row['Id']) + '\n')
                f.write(row['sequence'] + '\n')

    def cluster(self):
        args = [
                  "-c", str(self.similarity),
                  "-n", str(self.word_size),
                  "-aS", str(self.coverage),
                  "-s", str(self.length_cutoff),
                  "-g", "0",
                  "-r", "0",
                  "-M", "0",
                  "-l", str(self.word_size),
                  "-d", "0",
                  "-T", "0",
               ]

        subprocess.call(["cd-hit-est", "-i", str(self.fasta_path.resolve()),
                          "-o", str(self.ana_path.resolve())] + args)  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        parser = CDHITClusterParser(self.ana_clstr_path)
        cluster = parser.parse()
        cluster = get_scores_and_refs(cluster)
        assert not cluster.empty

        return cluster



class Infernal():
    def __init__(self,
                 working_dir: str = 'working_dir',
                 E: float = 0.001,
                 incE: float = 0.001,
                 aln_outfile = 'infernal.aln',
                 ):
        self.working_dir = str(Path(working_dir).resolve())
        self.aln = str(Path(working_dir, aln_outfile).resolve())
        self.E = E
        self.incE = incE

    def search_database(self,
                        cm_database: str,
                        identifier: str,
                        fasta_db: str,
                        ) -> list:

        call1 = ["echo", str(identifier)]
        call2 = ["cmfetch", "-f", f"{cm_database}", "-"]
        call3 = ["cmsearch", "-A", f"{self.aln}", "-E", f"{self.E}", "--incE", f"{self.incE}", "-", f"{fasta_db}"]

        ps1 = subprocess.Popen(call1, stdout=subprocess.PIPE)
        ps2 = subprocess.Popen(call2, stdin=ps1.stdout, stdout=subprocess.PIPE)
        subprocess.run(call3, stdin=ps2.stdout, stdout=subprocess.DEVNULL)

        hit_ids = stockholm2idlist(self.aln)
        return hit_ids

    def get_family_information(self,
                               queries_fasta_path : str,
                               cm_path : str,
                               clanin_path : str,
                               outpath : str,
                               ):
        subprocess.call(["cmpress", cm_path])
        subprocess.call(["cmscan", "--rfam", "--cut_ga", "--nohmmonly", "--oskip", "--tblout", outpath, "--fmt", "2", "--clanin", clanin_path, cm_path, queries_fasta_path])



    def search(self, cm_path: str, fasta_path: str, outfile: Optional[Union[str, Path]] = None):
        if outfile:
            subprocess.call(["cmsearch", "-A", f"{outfile}", "-E", f"{self.E}", "--incE", f"{self.incE}", cm_path, fasta_path])
            try:
                hit_ids = stockholm2idlist(outfile)
            except ValueError as e:
                print('Note:', 'No hits satisfy inclusion thresholds; no alignment saved')
                hit_ids = []
            except FileNotFoundError as e:
                print('Note:', 'no alignment file found')
                hit_ids = []
        else:
            subprocess.call(["cmsearch", "-A", f"{self.aln}", "-E", f"{self.E}", "--incE", f"{self.incE}", cm_path, fasta_path])
            try:
                hit_ids = stockholm2idlist(self.aln)
            except ValueError as e:
                print('Note:', 'No hits satisfy inclusion thresholds; no alignment saved')
                hit_ids = []
            except FileNotFoundError as e:
                print('Note:', 'no alignment file found')
                hit_ids = []

        return hit_ids

    def build_noss(self, stk_path: str, out_stem: str, name: str = '', cm_dir: str = ''):
        if not cm_dir:
            if not name:
                subprocess.call(["cmbuild", "--noss", f"{self.working_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "--noss", "-F", "-n", name, f"{self.working_dir}/{out_stem}.cm", stk_path])
        else:
            if not name:
                subprocess.call(["cmbuild", "--noss", f"{cm_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "--noss", "-F", "-n", name, f"{cm_dir}/{out_stem}.cm", stk_path])

    def build(self, stk_path: str, out_stem: str, name: str = '', cm_dir: str = ''):
        if not cm_dir:
            if not name:
                subprocess.call(["cmbuild", "-F", f"{self.working_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "-F", "-n", name, f"{self.working_dir}/{out_stem}.cm", stk_path])
        else:
            if not name:
                subprocess.call(["cmbuild", "-F", f"{cm_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "-F", "-n", name, f"{cm_dir}/{out_stem}.cm", stk_path])


    def calibrate(self, cm_path: str):
        subprocess.call(["cmcalibrate", cm_path])

    def sample_msa(self, cm_path : str, outpath : str, n : int = 1000, seed : int = 42):
        subprocess.call(["cmemit", "-o", outpath, "-a", "-N", str(n), "--seed", str(seed), cm_path])



class RNAAlifold():
    def __init__(self, working_dir: str = 'working_dir'):
        pass

    def fold(self, stk_path: str, aln_stk_prefix: str = 'test'):
        subprocess.call(["RNAalifold", stk_path, f"--aln-stk={aln_stk_prefix}"])


class MMSEQS2():
    def __init__(self, working_dir: str = 'working_dir'):
        self.working_dir = working_dir

    def easy_cluster_df(self,
                     df: pd.DataFrame,
                     fasta_file:str,
                     output_file_stem: str,
                     ):

        fasta_path = str(Path(self.working_dir, fasta_file).resolve())
        out = str(Path(self.working_dir, output_file_stem).resolve())

        df['Id'] = df['Id'].astype(str)
        df2fasta(df=df, out_path=fasta_path)

        subprocess.run(["mmseqs", "easy-cluster", fasta_path, out, self.working_dir])  # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        clusters_ids = pd.read_csv(str(Path(out).with_suffix('')) + '_cluster.tsv', sep='\t', names=['ref', 'member'])

        dfs = []
        for name, cluster in clusters_ids.groupby('ref'):
            l = [str(i) for i in cluster['member'].to_list()]
            t = df[df['Id'].isin(l)]
            t.loc[:, 'cluster'] = name
            t.loc[:, 'ref'] = t['Id'].apply(lambda x: x == name)
            dfs.append(t)

        return pd.concat(dfs)


class LocARNA():
    def __init__(self, working_dir: str = 'working_dir', prob : bool = True, consensus='alifold'):
        self.working_dir = working_dir
        self._prob = prob
        self.consensus = consensus

    def align_multi(self, fasta_file: str):
        if self._prob:
            if self.consensus:
                subprocess.call(["mlocarna", f"{fasta_file}", "--probabilistic", "--stockholm", '--consensus-structure', self.consensus], cwd=self.working_dir)
            else:
                subprocess.call(["mlocarna", f"{fasta_file}", "--probabilistic", "--stockholm"], cwd=self.working_dir)
        else:
            if self.consensus:
                subprocess.call(["mlocarna", f"{fasta_file}", "--stockholm", '--consensus-structure', self.consensus], cwd=self.working_dir)
            else:
                subprocess.call(["mlocarna", f"{fasta_file}", "--stockholm"], cwd=self.working_dir)

