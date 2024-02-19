import pandas as pd
import numpy as np

from Bio import SearchIO

from RnaBench.lib.data.parser.base_parser import BaseParser
# from rna_data.parser.base_parser import process_raw_alignment
import warnings
warnings.filterwarnings('ignore')

class BlastTabParser(BaseParser):
    def __init__(
                     self,
                     path,
                     query,
                     comments=True,
                     msa_max_size=100000,
                     max_hsps=None,
                 ):
        super().__init__(path, msa_max_size=msa_max_size)
        self.comments = comments
        self.query = query
        self.max_hsps = max_hsps


    def parse(self):
        try:
            self.qresults = SearchIO.read(self.path, 'blast-tab', comments=self.comments)
        except ValueError as v:
            print(v)
            return None

        if len(self.qresults) == 0:
            return None

        hit_list = []
        q_id = self.qresults.accession
        # for i, hit in enumerate(self.qresults):
        for hit in self.qresults:
            # t_id = hit.accession + f"-hit-{i}"
            t_id = hit.accession
            if self.max_hsps:
                max_hsps = self.max_hsps
            else:
                max_hsps = len(hit)
            for j, hsp in enumerate(hit[:self.max_hsps]):
                e_value = str(hsp.evalue)
                q_s, q_e = hsp.query_start, hsp.query_end
                t_s, t_e = hsp.hit_start, hsp.hit_end
                q_a = str(hsp.query.seq)
                t_a = str(hsp.hit.seq)
                q_align = self.query[:int(q_s)] + q_a + self.query[int(q_e):]
                q_align = q_align.upper().replace('T', 'U')
                t_align = '-' * int(q_s) + t_a + '-' * (len(self.query) - (int(q_e)))
                t_align = t_align.upper().replace('T', 'U')

                hit_list.append((q_id, t_id + f"-hsp-{j}", e_value, q_align, t_align, len(q_align)))
        # TODO: apply sorting for e_values and msa_max_size after dropping of duplicates!!!
        # hit_list = sorted(hit_list, key=lambda x: float(x[2]))
        msa = pd.DataFrame(hit_list, columns=['query_Id', 'Id', 'e_value', 'query_alignment', 'target_alignment', 'length'])
        msa.loc[:, 'sequence'] = msa['target_alignment'].apply(lambda x: x.replace('-', ''))
        msa.loc[:, 'query_sequence'] = msa['query_alignment'].apply(lambda x: x.replace('-', ''))
        # msa.rename({'Q_ID': 'query_Id', 'T_ID': 'Id', 'E':, 'Q_A', 'T_A', 'len'})
        # # msa = process_raw_alignment(ali_df, len(self.query))
        # # msa.loc[:, 'tool'] = 'blastn'
        # msa = msa.astype({'ID': np.str_, 'e_value': np.float64, 'sequence': np.str_, 'tool': np.str_})
        # msa = msa.sort_values(by=['e_value'], ascending=True)
        # msa = msa.astype({'ID': np.str_, 'e_value': np.str_, 'sequence': np.str_, 'tool': np.str_})
        # # msa = msa[:self.msa_max_size]
        return msa


if __name__ == '__main__':
    from RnaBench.core.utils import df2fasta
    import pandas as pd

    from pathlib import Path

    fasta_dir = '/home/fred/github/RNA-benchmark/fasta_files'
    # out_dir = '/home/fred/github/RNA-benchmark/alignment_files'
    tbl_dir = '/home/fred/github/RNA-benchmark/MSA_raw'

    for p in Path(fasta_dir).glob('*.fasta'):
        file_id = p.stem
        with open(Path(fasta_dir, f"{file_id}.fasta")) as f:
            lines = f.readlines()

        for line in lines:
            if not line.startswith('>'):
                query = line.strip()

        # print(query)
        query_df = pd.DataFrame({'Id': file_id, 'sequence': query}, index=[int(file_id)])

        tbl_path = Path(tbl_dir, f"{file_id}.tbl")
        tbl_path = str(tbl_path.resolve())

        tab_parser = BlastTabParser(tbl_path, query=query, comments=True)
        hits = tab_parser.parse()
        if hits is None:
            print('### No hits for Id', file_id)
            msa = query_df.copy()
        else:
            hits.drop_duplicates('sequence', inplace=True)
            msa = pd.concat([query_df, hits])
            print('### MSA of Id', file_id, 'has size', len(msa))

        # print(msa)
        msa.dropna(axis=1, inplace=True)

        # print(msa)

        df2fasta(msa, Path(tbl_dir, f"{file_id}_alignment.fasta"))

    # p = 'MSA_raw/102592.tbl'

    # parser = BlastTabParser(p, query='CACUGGUGCAAAUUUGCACUAGUCUAAAACUCCUCGAUUACAUACACAAAGCA', comments=True)

    # msa = parser.parse()


    # for i, row in msa.iterrows():
    #     print(row['sequence'])
    #     print(row['query_sequence'])
    #     # print(row)
    # for i, row in msa.iterrows():
    #     print(row['target_alignment'])
    #     print(row['query_alignment'])

    # print(msa.shape)
