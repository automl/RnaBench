import gdown
import subprocess

import pandas as pd

from pathlib import Path


DATASETS = {
  'intra_family_benchmark.plk.gz': 'https://drive.google.com/uc?id=1AN30C-y_gUoq5vAm7YYEdIubDvUNT5Dl',  # https://drive.google.com/file/d/1AN30C-y_gUoq5vAm7YYEdIubDvUNT5Dl/view?usp=drive_link
  'intra_family_train.plk.gz': 'https://drive.google.com/uc?id=1UJgyFyoQZPkdDP3gAIS07iNk7YiTz9dx',  # https://drive.google.com/file/d/117PFf3Dl_Cv63K0ciNDCtAOcr84XIiLn/view?usp=drive_link; https://drive.google.com/file/d/1UJgyFyoQZPkdDP3gAIS07iNk7YiTz9dx/view?usp=drive_link
  'intra_family_valid.plk.gz': 'https://drive.google.com/uc?id=1MVpXEn7_Gfr8-T7L9KRFjmvtdEEoxgO6',  # https://drive.google.com/file/d/1MVpXEn7_Gfr8-T7L9KRFjmvtdEEoxgO6/view?usp=drive_link
  'inter_family_benchmark.plk.gz': 'https://drive.google.com/uc?id=1DBbTSdmrsxZv5Tjjnq2yDMFYZwpJ-RS1',  # https://drive.google.com/file/d/1DBbTSdmrsxZv5Tjjnq2yDMFYZwpJ-RS1/view?usp=drive_link
  'inter_family_train.plk.gz': 'https://drive.google.com/uc?id=1fpP1S1oh1jTKlyDOb2TAeFxyCFxFXPXw',  # https://drive.google.com/file/d/1GqdvhwUFBJ21wiPB364oUoW21fphsqTn/view?usp=drive_link; https://drive.google.com/file/d/1fpP1S1oh1jTKlyDOb2TAeFxyCFxFXPXw/view?usp=drive_link
  'inter_family_valid.plk.gz': 'https://drive.google.com/uc?id=1QeS8WG9psIo1SOzNY4p-oVEw7v6hn3ww',  # https://drive.google.com/file/d/1QeS8WG9psIo1SOzNY4p-oVEw7v6hn3ww/view?usp=drive_link
  'inter_family_fine_tuning_train.plk.gz': 'https://drive.google.com/uc?id=1lit7xeqpqPrjdyz-IM7k_GIfxyuW_cYb',  # https://drive.google.com/file/d/1ZWVM49pnCSbU8qePFQyrKdHIpByaEBIi/view?usp=drive_link; https://drive.google.com/file/d/1lit7xeqpqPrjdyz-IM7k_GIfxyuW_cYb/view?usp=drive_link
  'biophysical_model_benchmark.plk.gz': 'https://drive.google.com/uc?id=1pCg6iPRJkldlqbE1sU9_vO5xTfQ4N0Kc',  # https://drive.google.com/file/d/1pCg6iPRJkldlqbE1sU9_vO5xTfQ4N0Kc/view?usp=drive_link
  'biophysical_model_train.plk.gz': 'https://drive.google.com/uc?id=1ez4uXnE4SvlPHjduVrcKgc8L-py7Pr8v',  # https://drive.google.com/file/d/1ez4uXnE4SvlPHjduVrcKgc8L-py7Pr8v/view?usp=drive_link
  'biophysical_model_valid.plk.gz': 'https://drive.google.com/uc?id=1XIYOnpNjT38C3Q1fIvlaw62n7A8YmPZT',  # https://drive.google.com/file/d/1XIYOnpNjT38C3Q1fIvlaw62n7A8YmPZT/view?usp=drive_link
  'riboswitch_design_train.plk.gz': 'https://drive.google.com/uc?id=1QKSovEar70XLGevQgEtlvjOJAPIBBwh4',    # https://drive.google.com/file/d/1QKSovEar70XLGevQgEtlvjOJAPIBBwh4/view?usp=drive_link
  'inverse_rna_folding_benchmark.plk.gz': 'https://drive.google.com/uc?id=1mSH_upJHBlC9XoUcMVH1MucyVLt4Rhgx',  # https://drive.google.com/file/d/1hlcKaPLjgfdZB2SfvjEmCO6qcK3DC65e/view?usp=drive_link; https://drive.google.com/file/d/1mSH_upJHBlC9XoUcMVH1MucyVLt4Rhgx/view?usp=drive_link
  'inverse_rna_folding_train.plk.gz': 'https://drive.google.com/uc?id=1baVwuEWJnBlFJWrypWE6AgPJzP2reUYw',  # https://drive.google.com/file/d/1dj1SWCJ3bxHge__yrD6VH2apGAHAOWff/view?usp=drive_link; https://drive.google.com/file/d/1_BIrEx19MQINIB2Ge3LabUyJMqcR_k6H/view?usp=drive_link; https://drive.google.com/file/d/1baVwuEWJnBlFJWrypWE6AgPJzP2reUYw/view?usp=drive_link
  'inverse_rna_folding_valid.plk.gz': 'https://drive.google.com/uc?id=1-3UBqU4WPvH_hEJTaekx_U7QAMcSLKW9',  # https://drive.google.com/file/d/1A6Tsoh8mJM9c1gho8_h5wOesbWVpguVZ/view?usp=drive_link; https://drive.google.com/file/d/1-3UBqU4WPvH_hEJTaekx_U7QAMcSLKW9/view?usp=drive_link
  'constrained_design_benchmark.plk.gz': 'https://drive.google.com/uc?id=1ZU_VKBveT25GMuEyMxK6_F9DcMp_N1bY',  # https://drive.google.com/file/d/1I9o3QDHWYG3GjrqzPlpw7h_nWA-ld2Zg/view?usp=drive_link; https://drive.google.com/file/d/1ZU_VKBveT25GMuEyMxK6_F9DcMp_N1bY/view?usp=drive_link
  'constrained_design_train.plk.gz': 'https://drive.google.com/uc?id=12irVqG4al3UaYdfEQArQoKOChUhJGGus',  # https://drive.google.com/file/d/1UDZ8uHpbcRXZla2opQFa0D8eOuQs_9-i/view?usp=drive_link; https://drive.google.com/file/d/1NsmmviFcJ86EdsC9eE5TVL6Nm5nS5oks/view?usp=drive_link; https://drive.google.com/file/d/12irVqG4al3UaYdfEQArQoKOChUhJGGus/view?usp=drive_link
  'constrained_design_valid.plk.gz': 'https://drive.google.com/uc?id=1MeXC2E5Norf2rT4yXbeSo9xQB4NP2IrN',  # https://drive.google.com/file/d/1FXksh4E789mjQWYOmhoRN-nxt_km1j72/view?usp=drive_link; https://drive.google.com/file/d/1MeXC2E5Norf2rT4yXbeSo9xQB4NP2IrN/view?usp=drive_link
  'ArchiveII_pk.plk.gz': 'https://drive.google.com/uc?id=1W8hbU03lCdttuh5zKXPlU5aCHdk8p2aY',  # https://drive.google.com/file/d/1W8hbU03lCdttuh5zKXPlU5aCHdk8p2aY/view?usp=drive_link
  'anta_pseudo-test.plk.gz': 'https://drive.google.com/uc?id=1ORh9HUJkZvMg2UoufvrEoXOpn0JceUS_',  # https://drive.google.com/file/d/1ORh9HUJkZvMg2UoufvrEoXOpn0JceUS_/view?usp=drive_link
  'anta_rfam-test.plk.gz': 'https://drive.google.com/uc?id=1Fpl1reKaLejgVbJbMxqrmDDj7NsWF3zD',  # https://drive.google.com/file/d/1Fpl1reKaLejgVbJbMxqrmDDj7NsWF3zD/view?usp=drive_link
  'anta_strand-test.plk.gz': 'https://drive.google.com/uc?id=1JnRJIci07h_OptZGgwkzu8HsCxQw567G',  # https://drive.google.com/file/d/1JnRJIci07h_OptZGgwkzu8HsCxQw567G/view?usp=drive_link
  'rfam_learn-test.plk.gz': 'https://drive.google.com/uc?id=1JlFIfljDd8pvNfKkPyX8di9Qzpbp-GvB',  # https://drive.google.com/file/d/1JlFIfljDd8pvNfKkPyX8di9Qzpbp-GvB/view?usp=drive_link
  'rfam_taneda-test.plk.gz': 'https://drive.google.com/uc?id=1D1_0uGM-z1pASHKtdnTVqj8fzCxlsh72',  # https://drive.google.com/file/d/1D1_0uGM-z1pASHKtdnTVqj8fzCxlsh72/view?usp=drive_link
  'eterna100_v1.plk.gz': 'https://drive.google.com/uc?id=1ciA2vIFZ10ukAeZ88-sRqpTHyfutfgjD',  # https://drive.google.com/file/d/1ciA2vIFZ10ukAeZ88-sRqpTHyfutfgjD/view?usp=drive_link
  'eterna100_v2.plk.gz': 'https://drive.google.com/uc?id=1Iry9e_VFsE_hXLAlIuycV1VpmfnF49rJ',  # https://drive.google.com/file/d/1Iry9e_VFsE_hXLAlIuycV1VpmfnF49rJ/view?usp=drive_link
  'initial_data_benchmark.plk.gz': 'https://drive.google.com/uc?id=14_voPJxbCH65oENamZKizNw0U4IEiJSQ',  # https://drive.google.com/file/d/1GcVD2x-zyGBV-RnZB4eAPsL4dfQiRhtb/view?usp=drive_link; https://drive.google.com/file/d/14_voPJxbCH65oENamZKizNw0U4IEiJSQ/view?usp=drive_link
  'pdb_ts_hard.plk.gz': 'https://drive.google.com/uc?id=1goXH05N2OcVi4LED05OA-EpWV_NuVHmw',  # https://drive.google.com/file/d/1goXH05N2OcVi4LED05OA-EpWV_NuVHmw/view?usp=drive_link
  'pdb_vl1.plk.gz': 'https://drive.google.com/uc?id=1KUMn_8IhokrHz2YXRnECCtOST-gZGNs7',  # https://drive.google.com/file/d/1KUMn_8IhokrHz2YXRnECCtOST-gZGNs7/view?usp=drive_link
  'pdb_ts1.plk.gz': 'https://drive.google.com/uc?id=1zTIwSqixGcWBhsRz4rup_K-a2ERK62Ty',  # https://drive.google.com/file/d/1zTIwSqixGcWBhsRz4rup_K-a2ERK62Ty/view?usp=drive_link
  'pdb_ts2.plk.gz': 'https://drive.google.com/uc?id=1OQLkNlOqFBpA7YLfLJqKL-Xy1qO8M1V3',  # https://drive.google.com/file/d/1OQLkNlOqFBpA7YLfLJqKL-Xy1qO8M1V3/view?usp=drive_link
  'pdb_ts3.plk.gz': 'https://drive.google.com/uc?id=1E-fCrUvR-C12Q7KMlWHDz-PzJkS2LX67',  # https://drive.google.com/file/d/1E-fCrUvR-C12Q7KMlWHDz-PzJkS2LX67/view?usp=drive_link
  'bprna_vl0.plk.gz': 'https://drive.google.com/uc?id=1Ymjcp6YQuuVuJpuIgZy1JlCjV58d524n',  # https://drive.google.com/file/d/1Ymjcp6YQuuVuJpuIgZy1JlCjV58d524n/view?usp=drive_link
  'bprna_ts0.plk.gz': 'https://drive.google.com/uc?id=1QOx5d5NDhCw3Lkvbx4JoH4FjkFgfnHnF',  # https://drive.google.com/file/d/1QOx5d5NDhCw3Lkvbx4JoH4FjkFgfnHnF/view?usp=drive_link
  'biophysical_model_initial_test.plk.gz': 'https://drive.google.com/uc?id=1DRsG36qdY1-T3XG0HIa1wKIGaQNWmelr',  # https://drive.google.com/file/d/1DRsG36qdY1-T3XG0HIa1wKIGaQNWmelr/view?usp=drive_link
  'biophysical_model_initial_valid.plk.gz': 'https://drive.google.com/uc?id=1Lsv4MG5gHI0sj-cuqaaVrrfJkX2zbfJI',  # https://drive.google.com/file/d/1Lsv4MG5gHI0sj-cuqaaVrrfJkX2zbfJI/view?usp=drive_link
  'biophysical_model_initial_train.plk.gz': 'https://drive.google.com/uc?id=1dDfLY6RNhblFj7wtavAPdoKWqZ6zbiNu',  # https://drive.google.com/file/d/1dDfLY6RNhblFj7wtavAPdoKWqZ6zbiNu/view?usp=drive_link
}

def select_and_download(task, save_dir, all_data=True):
    if all_data:
        for s in ['_train', '_valid', '_benchmark']:
            try:
                download_single_file_from_gdrive(url=DATASETS[f"{task}{s}.plk.gz"],
                                                 destination=save_dir + '/' + f"{task}{s}.plk.gz")
            except Exception as e:
                pass
    else:
        try:
            download_single_file_from_gdrive(url=DATASETS[f"{task}_benchmark.plk.gz"],
                                             destination=save_dir + '/' + f"{task}_benchmark.plk.gz")
        except Exception as e:
            pass


def download_single_file_from_gdrive(url, destination):
    gdown.download(url, destination, quiet=False)


def download_rfam_cms(destination):
    subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.clanin"], cwd=destination)
    subprocess.call(["wget", "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"], cwd=destination)
    subprocess.call(["gunzip", "Rfam.cm.gz"], cwd=destination)
    subprocess.call(["cmpress", "Rfam.cm"], cwd=destination)


def download_3d_data(
                     destination : str,
                     resolution : str = '3.5',  # available options: 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 20.0, all
                     data_type : str = 'cif',  # available options: cif, fasta, pdb
                     bgsu_version : float = '3.286',
                     rna_type : str = 'solo',  # available options: 'solo' -> RNA only data; 'all' -> all molecules; 'complex' -> RNA-protein complexes; 'hybrid' -> DNA-RNA hybrids
                     representatives_only : bool = False,  # if True: download only representative members of every RNA equivalence class
                     ):
    rep = 'representative' if representatives_only  else "member"
    bgsu = str(bgsu_version).replace('.', '_')
    resolution = resolution.replace('.', '_')

    download_link = f'https://rnasolo.cs.put.poznan.pl/media/files/zipped/bunches/{data_type}/{rna_type}_{rep}_{data_type}_{resolution}__{bgsu}.zip'
    file_id = f"{rna_type}_{rep}_{data_type}_{resolution}__{bgsu}.zip"
    destination = f"{destination}/{rna_type}_{rep}_{data_type}_{resolution}__{bgsu}"
    Path(destination).mkdir(exist_ok=True, parents=True)

    subprocess.call(["wget", download_link], cwd=destination)
    # subprocess.call(["unzip", file_id], cwd=destination)
    # subprocess.call(["rm", file_id], cwd=destination)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_types', type=str, default=['2D', 'CMs'], nargs="+", help='Define which data to download')  # also available: 'rnasolo'
    parser.add_argument('--bgsu', default=3.286, type=float, help='download all 3D rna data from RNAsolo')
    parser.add_argument('--data_outdir', default='data', type=str, help='output_directory for 2D data')
    parser.add_argument('--rfam_cm_outdir', default='RnaBench/lib/data/CMs', type=str, help='output_directory for Rfam.cm')

    args = parser.parse_args()
    if '2D' in args.data_types:
        print('### Downloading all secondary structure data.')
        Path(args.data_outdir).mkdir(exist_ok=True, parents=True)
        for k, v in DATASETS.items():
            destination = f'{args.data_outdir}/{k}'
            if not Path(destination).is_file():
                url = v
                download_single_file_from_gdrive(url, destination)
            else:
                print(f'File {destination} already exists. Skipping.')
        print('### Download Rfam family CMs')
    if 'CMs' in args.data_types:
        Path(args.rfam_cm_outdir).mkdir(exist_ok=True, parents=True)
        download_rfam_cms(args.rfam_cm_outdir)
    if 'rnasolo' in args.data_types:
        print('### Download 3D data')
        bgsu = args.bgsu
        threedee_destination = Path(args.data_outdir, '3D')
        threedee_destination.mkdir(exist_ok=True, parents=True)
        res = ['1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '20.0', 'all']
        rep = [True, False]
        d_list = [[r, rep[0]] for r in res] + [[r, rep[1]] for r in res]
        for r, rep in d_list:
            try:
                download_3d_data(str(threedee_destination.resolve()),
                representatives_only=rep,
                resolution=r,
                bgsu_version=bgsu,
                )
            except Exception as e:
                print('### 3D data download not working. Try to increase the BGSU version via --bgsu.', f"Current version is {bgsu}")
                continue













