import pandas as pd
import numpy as np

def process_raw_alignment(df, length_threshold):
    gaps = set()

    insert_df = df.copy()
    insert_df.loc[:, 'gaps'] = np.nan

    for i, row in df[df['len'] > length_threshold].iterrows():
        current_gaps = get_gap_ids(row['Q_A'])
        gaps = gaps | current_gaps
        i_gaps = gaps - current_gaps

        insert_df.loc[i, 'gaps'] = [current_gaps]

    insert_df['gaps'] = insert_df['gaps'].apply(lambda x: x if isinstance(x, set) else set())

    df['T_A'] = insert_df.apply(lambda x: insert_gaps_target(x, gaps), axis=1)
    df['Q_A'] = insert_df.apply(lambda x: insert_gaps_query(x, gaps), axis=1)

    q_df = df.iloc[0][['Q_ID', 'Q_A']]
    q_df.rename(index={'Q_ID': 'ID', 'Q_A': 'sequence'}, inplace=True)
    q_df['e_value'] = '0.0'

    t_df = df[['T_ID', 'E', 'T_A']].copy()
    t_df.rename(columns={'T_ID': 'ID', 'E': 'e_value', 'T_A': 'sequence'}, inplace=True)
    t_df = t_df.drop_duplicates(subset=['sequence'])

    ali_df = pd.concat([q_df.to_frame().T, t_df])

    return ali_df

def get_gap_ids(seq):
    return set(i for i, s in enumerate(seq) if s == '-')

def insert_gaps_target(row, gaps):
    target = list(row['T_A'])
    i_gaps = gaps - row['gaps']
    for i in sorted(i_gaps):
        target.insert(i, '-')
    return ''.join(target)

def insert_gaps_query(row, gaps):
    query = list(row['Q_A'])
    i_gaps = gaps - row['gaps']
    for i in sorted(i_gaps):
        query.insert(i, '-')
    return ''.join(query)



class BaseParser():
    def __init__(
                     self,
                     path,
                     msa_max_size=100000,
                ):
        self.path = path
        self.msa_max_size = msa_max_size


    def parse(self):
        pass


    def save_msa(self):
        pass
