# This script requires
# - DSSR available via https://x3dna.org/
# - bpRNA available via https://github.com/hendrixlab/bpRNA



import pickle
import argparse

from pathlib import Path

from RnaBench.lib.data.pre_processing import PDB2DPipeline


parser = argparse.ArgumentParser()

parser.add_argument('--mmcif_dir', default='data/3D/solo_representative_cif_1_5__3_286', type=str, help='The Path to the .cif PDB files')  # all_representative_cif_all__3_298
parser.add_argument('--dssr_dir', default='/home/fred/research/DSSR', type=str, help='Path to the DSSR binaries')
parser.add_argument('--bprna_dir', default='external_algorithms/bpRNA', type=str, help='Path to the DSSR binaries')
parser.add_argument('--working_dir', default='working_dir', type=str, help='Path to the DSSR binaries')

args = parser.parse_args()

Path(args.working_dir).mkdir(exist_ok=True, parents=True)


pipeline = PDB2DPipeline(
  mmcif_dir = args.mmcif_dir,
  dssr_dir = args.dssr_dir,
  bprna_dir = args.bprna_dir,
  working_dir = args.working_dir,
)

df = pipeline.process_mmcif_dir()

with open('pdb_all_05_09_2023_rna_solo_bgsu_3_298_16:55.plk', 'wb') as f:
    pickle.dump(df, f)


