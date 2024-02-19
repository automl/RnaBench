import pandas as pd

from typing import Union
from pathlib import Path


class InfernalTbloutParser():
    """
    Parse the output of a tabular file resulting from applying Infernal to search
    the rfam database to get family information for a given set of query sequences.


    """
    def __init__(self,
                 tblout_path : Union[str, Path],
                 ):
        self._tbl_path = tblout_path

    def parse(self):
        family_info = []

        with open(self._tbl_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            line = [l for l in line.split() if l]

            idx, target_name, t_accession, query_name, q_accession, clan_name, mdl, \
            mdl_from, mdl_to, seq_from, seq_to, strand, trunc, pass_, gc, bias, \
            score, e_value, inc, olp, anyidx, afrct1, afrct2, winidx, wfrct1, \
            wfrct2 = line[:26]

            description = ' '.join(line[26:])

            hit_info = {
              'query_Id': query_name,
              'q_accession': q_accession,
              'target_name': target_name,
              't_accession': t_accession,
              'description': description,
              'e_value': e_value,
            }

            family_info.append(hit_info)

        return pd.DataFrame(family_info)


if __name__ == '__main__':
    tbl = 'data/data1_rfam.tblout'
    parser = InfernalTbloutParser(tblout_path=tbl)
    info = parser.parse()
    print(info)