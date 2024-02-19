import subprocess
import forgi
import pandas as pd

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from typing import List, Dict, Optional, Union
from pathlib import Path
from rna_tools.tools.rna_x3dna.rna_x3dna import x3DNA

class MMCIF3DParser():
    """
    Obtain 3D information from mmcif files.
    """
    def __init__(self,
                 mmcifdir : str,
                ):
        self._mmcifdir : str = mmcifdir
        self._parser : Bio.PDB.MMCIFParser = MMCIFParser()

    def mmcif2structure(self, file_path : str):
        file_name = str(Path(file_path).stem)
        return self._parser.get_structure(file_name, file_path)

    def parse_dir2structure(self):
        return [self.mmcif2structure(p) for p in Path(self._mmcifdir).glob('*.cif')]

    def mmcif2dict(self, file_path : str) -> Dict[str, Union[float, str, int]]:
        return MMCIF2Dict(file_path)

    def parse_dir2dict(self) -> List[Dict[str, Union[float, str, int]]]:
        return [self.mmcif2dict(p) for p in Path(self._mmcifdir).glob('*.cif')]

    def parse_dir2df(self) -> pd.DataFrame:
        dfs = []
        for i, p in enumerate(Path(self._mmcifdir).glob('*.cif'), 1):
            mmcif_dict = self.mmcif2dict(p)

            atoms = []
            for n_type, a_type, a_pos, a_sym, a_id, occ, equiv, chain, seq_pos, x, y, z in zip(mmcif_dict["_atom_site.label_comp_id"],
                                                                                               mmcif_dict["_atom_site.group_PDB"],
                                                                                               mmcif_dict["_atom_site.id"],
                                                                                               mmcif_dict["_atom_site.type_symbol"],
                                                                                               mmcif_dict["_atom_site.label_atom_id"],
                                                                                               mmcif_dict["_atom_site.occupancy"],
                                                                                               mmcif_dict["_atom_site.B_iso_or_equiv"],
                                                                                               mmcif_dict["_atom_site.label_asym_id"],
                                                                                               mmcif_dict["_atom_site.label_seq_id"],
                                                                                               mmcif_dict["_atom_site.Cartn_x"],
                                                                                               mmcif_dict["_atom_site.Cartn_y"],
                                                                                               mmcif_dict["_atom_site.Cartn_z"]):
                d = {
                  "nucleotide": n_type,
                  "atom_type": a_type,
                  "atom_position": int(a_pos),
                  "atom_symbol": a_sym,
                  "atom_id": a_id,
                  "x_coordinate": float(x),
                  "y_coordinate": float(y),
                  "z_coordinate": float(z),
                  "occupancy": float(occ),
                  "B_iso_or_equiv": float(equiv),
                  "chain": chain,
                  "sequence_position": int(seq_pos),
                }
                atoms.append(d)

            df = pd.DataFrame(atoms)
            df.loc[:, 'Id'] = i
            df.loc[:, "solo_accession"] = str(p.stem)
            # print(df)
            dfs.append(df)

        return pd.concat(dfs)


class MMCIF2DParser():
    """
    Obtain 2D information from PDB data in mmcif format.

    Note:
    * Requires a working version of DSSR available via
        https://columbia.resoluteinnovation.com/
    * Requires bpRNA available via
        https://github.com/hendrixlab/bpRNA
    """
    def __init__(self, file_type : str = 'mmcif', PERMISSIVE : int = 1):
        if file_type == 'mmcif':
            self._parser : Bio.PDB.MMCIFParser = MMCIFParser()  # PERMISSIVE=PERMISSIVE
        elif file_type == 'pdb':
            self._parser : Bio.PDB.PDBParser = PDBParser()  # PERMISSIVE=PERMISSIVE
        else:
            raise UserWarning(f"Unknown PDB file format: {file_type} provided for 2D parser.")


    def read_structure(self, mmcif_path : Path):
        self.structure = self._parser.get_structure(mmcif_path.stem, mmcif_path)

    def get_header(self):
        return self.structure.header

    def get_rna_chain_ids(self):
        self.chains = [c.get_id() for c in self.structure.get_chains() if forgi.threedee.utilities.pdb.contains_rna(c)]

    def get_sequence(self, mmcif_path : Path):
        s = subprocess.call(["rna_pdb_tools.py", "--get-seq", str(mmcif_path.resolve())])



if __name__ == '__main__':
    import pickle
    # mmcif_dir = "/home/fred/github/RNA-benchmark/data/solo_member_cif_3_5__3_265"
    mmcif_dir = "data/3D/solo_representative_cif_1_5__3_284"

    parser = MMCIF3DParser(mmcif_dir)
    df = parser.parse_dir2df()
    print(df)
    print(df.groupby(['solo_accession', 'chain']).count())

    # with open(mmcif_dir + '.plk', 'wb') as f:
    #     pickle.dump(df, f)

