import pandas as pd

from collections import defaultdict
from pathlib import Path

from RnaBench.lib.data.parser.base_parser import BaseParser


class CDHITClusterParser(BaseParser):
    def __init__(self, path):
        super().__init__(path=path)


    def parse(self):
        clusters = []
        with open(self.path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if 'Cluster' in line:
                    current_cluster = line.split()[-1]
                else:
                    s_id = line.split()[2][1:-3]
                    score = line.split()[-1]
                    clusters.append((current_cluster, s_id, score))
        df = pd.DataFrame(clusters, columns=['cluster', 'Id', 'sim'])
        return df
