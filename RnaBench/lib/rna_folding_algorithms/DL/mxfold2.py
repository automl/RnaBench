import subprocess

from pathlib import Path

from RnaBench.lib.utils import db2pairs

class MxFold2():
    def __init__(self,
                 working_dir='working_dir',
                 gpu=-1,
                 model=None,  # currently no other models implemented
                 ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True, parents=True)
        self.gpu = gpu
        self.model = model

    def __name__(self):
        return 'MXFold2'

    def __repr__(self):
        return 'MXFold2'

    def __call__(self, sequence, id=0):
        fasta_path = Path(self.working_dir, 'mxfold2in.fasta')

        with open(fasta_path, 'w+') as f:
            f.write('>' + str(id) + '\n')
            f.write(''.join(sequence)+'\n')

        intermediate_output = Path(self.working_dir, 'mxfold2_predictions.out')

        infer_mxfold2(intermediate_output=intermediate_output,
                      fasta_path=fasta_path,
                      model=self.model,
                      gpu=self.gpu,
                      )
        pairs = parse_intermediate_output(intermediate_output)
        fasta_path.unlink()
        intermediate_output.unlink()
        return pairs


def infer_mxfold2(intermediate_output, fasta_path, model=None, gpu=-1):
    with open(intermediate_output, 'wb+') as f:
        if model:
            model = f"@./thirdparty_algorithm/MXFold2/models/{model}.conf"
            subprocess.call(["mxfold2", "predict", model, fasta_path, "--gpu", str(gpu)], stdout=f)
        else:
            subprocess.call(["mxfold2", "predict", fasta_path, "--gpu", str(gpu)], stdout=f)


def parse_intermediate_output(intermediate_output):
    with open(intermediate_output, 'rb') as f:
        lines = [line.decode('utf-8').strip() for line in f.readlines()]

    # print(lines)

    for i, line in enumerate(lines):
        if i % 3 == 2:
            pred = line.split()[0]
    pairs = db2pairs(pred)
    return pairs
