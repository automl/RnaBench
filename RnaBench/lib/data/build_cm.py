import argparse
import subprocess
import RNA
import pandas as pd

from pathlib import Path

from RnaBench.lib.data.similarity import LocARNA, Infernal, RNAAlifold
from RnaBench.lib.data.parser.blast_tab_parser import BlastTabParser
from RnaBench.lib.utils import df2fasta

parser = argparse.ArgumentParser()
parser.add_argument('--df_path', type=str, help='Path to dataframe pickle file')
parser.add_argument('--tbl_dir', default='working_dir/MSA_raw', type=str, help='ncbi blastn raw output dir')
parser.add_argument('--fasta_dir', default='working_dir/fasta_files', type=str, help='fasta file directory')
parser.add_argument('--ncbi_blastn_bin_dir', default='external_algorithms/ncbi-blast-2.12.0+/bin/', type=str, help='bin directory for blastn')
parser.add_argument('--locarna_output_dir', default='working_dir/locarna_alignments', type=str, help='alignment output directory')
parser.add_argument('--alifold_output_dir', default='working_dir/alifold', type=str, help='alifold output directory')
parser.add_argument('--cm_out_dir', default='CMs', type=str, help='CM output directory')
parser.add_argument('--e_value', default=0.1, type=float, help='infernal e_value')
# parser.add_argument('--file_id', type=str, help='file id to create alignment for')

args = parser.parse_args()

df = pd.read_pickle(args.df_path)

Path(args.locarna_output_dir).mkdir(exist_ok=True, parents=True)
Path(args.cm_out_dir).mkdir(exist_ok=True, parents=True)
Path(args.fasta_dir).mkdir(exist_ok=True, parents=True)
Path(args.alifold_output_dir).mkdir(exist_ok=True, parents=True)
Path(args.tbl_dir).mkdir(exist_ok=True, parents=True)

locarna = LocARNA(working_dir=args.locarna_output_dir)
rna_ali_fold = RNAAlifold(working_dir=args.alifold_output_dir)

print('### Start Building CMs for', len(df), 'samples')

for i, row in df.iterrows():
    file_id = row['Id']
    print('### process sequence ID', file_id)
    infernal = Infernal(working_dir=args.cm_out_dir,
                        E=args.e_value,
                        incE=args.e_value,
                        aln_outfile=f"{file_id}.aln",
                        )
    with open(Path(args.fasta_dir, f"{file_id}.fasta"), 'w+') as f:
        f.write('>'+str(file_id)+'\n')
        f.write(''.join(row['sequence'])+'\n')
    print('### Query NCBI nt database for sequence\n', ''.join(row['sequence']))
    if not Path(args.tbl_dir, f"{file_id}.tbl").is_file():
        subprocess.call(['./blastn', '-db', 'nt', '-query', str(Path(args.fasta_dir, f"{file_id}.fasta").resolve()), '-out', str(Path(args.tbl_dir, f"{file_id}.tbl").resolve()), '-evalue', str(args.e_value), '-outfmt', '7 qacc sacc pident nident score evalue bitscore qstart qend sstart send qseq sseq ', '-remote'], cwd=args.ncbi_blastn_bin_dir)
    else:
        print('### BLASTN .tbl file already exists. Continue with building CM.')

    try:
        with open(Path(args.fasta_dir, f"{file_id}.fasta")) as f:
            lines = f.readlines()

        for line in lines:
            if not line.startswith('>'):
                query = line.strip()

        print(query)

        tbl_path = Path(args.tbl_dir, f"{file_id}.tbl")
        tbl_path = str(tbl_path.resolve())
        print('### Parse BLASTN output')
        tab_parser = BlastTabParser(tbl_path, query=query, comments=True)
        hits = tab_parser.parse()
        hits.drop_duplicates('sequence', inplace=True)
        query_df = pd.DataFrame({'Id': file_id, 'sequence': query}, index=[int(file_id)])
        msa = pd.concat([query_df, hits])


        msa.dropna(axis=1, inplace=True)
        print('### MSA:')
        print(msa)

        df2fasta(msa, Path(args.locarna_output_dir, f"{file_id}_alignment.fasta"))

        print('### Try to get alignment from LocARNA')
        locarna.align_multi(f"{file_id}_alignment.fasta")

        locarna_res_dir = Path(args.locarna_output_dir, f"{file_id}_alignment.out")
        alignment_path = Path(locarna_res_dir, 'results', 'result.stk')

    except Exception as e:
        print('### No matching sequences found')
        print('### Get consensus structure with RNAFold and continue')
        structure, energy = RNA.fold(''.join(row['sequence']))
        alignment_path = Path(args.locarna_output_dir, f"{file_id}_alignment_rnafold.out")

        with open(alignment_path, 'w+') as f:
            f.write('# STOCKHOLM 1.0'+'\n')
            f.write('\n')
            f.write(str(file_id)+'          '+''.join(row['sequence'])+'\n')
            f.write('#=GC SS_cons'+'          '+structure+'\n')
            f.write('//'+'\n')

    print('### Build CM with Infernal')
    infernal.build(stk_path=str(alignment_path.resolve()),
                   out_stem=f"{file_id}",
                   name=f"{file_id}",
                   cm_dir=args.cm_out_dir)
    print('### Start CM caibration')
    infernal.calibrate(cm_path=str(Path(args.cm_out_dir, f"{file_id}.cm").resolve()))




