import pathlib
import subprocess
from pathlib import Path
import torch
import sys
import os
import numpy as np
import urllib.request
# Get the current directory of the Python file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(grandparent_dir)

from RnaBench.lib.utils import db2pairs
from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.utils.configuration import Config
from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.model.RNAformer import RiboFormer
from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.pl_module.datamodule_rna import IGNORE_INDEX, PAD_INDEX
from RnaBench.lib.rna_folding_algorithms.DL.RNAformer.utils.data.rna import CollatorRNA


NUCS = {
    'T': 'U',
    'P': 'U',
    'R': 'A',  # or 'G'
    'Y': 'C',  # or 'T'
    'M': 'C',  # or 'A'
    'K': 'U',  # or 'G'
    'S': 'C',  # or 'G'
    'W': 'U',  # or 'A'
    'H': 'C',  # or 'A' or 'U'
    'B': 'U',  # or 'G' or 'C'
    'V': 'C',  # or 'G' or 'A'
    'D': 'A',  # or 'G' or 'U'
    'N': 'N',
    'A': 'A',
    'U': 'U',
    'C': 'C',
    'G': 'G',
}


class RNAformer():
    def __init__(self,
                 working_dir='working_dir',
                 gpu=-1,
                 checkpoint_dir='RnaBench/lib/rna_folding_algorithms/DL/RNAformer/checkpoints',
                 dim=128,  # available options: 256, 128
                 cycling=False,  # True only for dim=256
                 max_len=None,
                 ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True, parents=True)
        self.gpu = gpu

        if not cycling:
            self.model_name = f"ts0_conform_dim{dim}_32bit"
        else:
            self.model_name = f"ts0_conform_dim{dim}_cycling_32bit"

        self.checkpoint = Path(checkpoint_dir, self.model_name)

        if not self.checkpoint.is_dir() or not Path(self.checkpoint, 'config.yml').is_file() or not Path(self.checkpoint, 'state_dict.pth').is_file():
            self.checkpoint.mkdir(exist_ok=True, parents=True)
            print("Downloading model checkpoints")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{self.model_name}/config.yml",
                f"{self.checkpoint}/config.yml")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{self.model_name}/state_dict.pth",
                f"{self.checkpoint}/state_dict.pth")
            print('Download successful')
        
        model_dir = self.checkpoint  # 'RnaBench/lib/rna_folding_algorithms/DL/RNAformer/checkpoints/ts0_conform_dim64_32bit'
        config_dir = self.checkpoint  # 'RnaBench/lib/rna_folding_algorithms/DL/RNAformer/checkpoints/ts0_conform_dim64_32bit'

        model_dir = pathlib.Path(model_dir)
        config_dir = pathlib.Path(config_dir)
        config = Config(config_file=config_dir / 'config.yml')
        state_dict = torch.load(model_dir / 'state_dict.pth', map_location=torch.device('cpu'))
    
        if config.trainer.precision == 'bf16':
            state_dict = {k: v.bfloat16() for k, v in state_dict.items()}
        elif config.trainer.precision == 'fp16' or config.trainer.precision == '16':
            state_dict = {k: v.half() for k, v in state_dict.items()}
        else:
            state_dict = {k: v.float() for k, v in state_dict.items()}
            config.RNAformer.flash_attn = False
    
        self.max_len = state_dict["seq2mat_embed.src_embed_1.embed_pair_pos.weight"].shape[1]
    
        model_config = config.RNAformer
        model_config.seq_vocab_size = 5
        if max_len:
            model_config.max_len = max_len
        else:
            model_config.max_len = self.max_len
    
        self.model = RiboFormer(model_config)
    
        self.model.load_state_dict(state_dict, strict=True)
    
        # model = model.cuda()
    
        self.ignore_index = IGNORE_INDEX
        self.pad_index = PAD_INDEX
    
        self.collator = CollatorRNA(self.pad_index, self.ignore_index)

        self.seq_vocab = ['A', 'C', 'G', 'U', 'N']
        self.seq_stoi = dict(zip(self.seq_vocab, range(len(self.seq_vocab))))


    def __name__(self):
        return 'RNAformer_' + self.model_name

    def __repr__(self):
        return 'RNAformer_' + self.model_name

    def __call__(self, sequence, id=0):
        model = self.model.eval()

        sequence = [NUCS[i] for i in sequence]

        # pairs = infer_rnaformer(model_dir=model_dir, config_dir=config_dir, sequence=sequence, gpu=self.gpu,)
        length = len(sequence)
        mean_triual = True
        
        # print(sequence)

        int_sequence = list(map(self.seq_stoi.get, sequence))
        # print(int_sequence)
        input_sample = torch.LongTensor(int_sequence)
    
        input_sample = {'src_seq': input_sample, 'length': torch.LongTensor([len(input_sample)])[0]}
        batch = self.collator([input_sample])
        with torch.no_grad():
            # logits, mask = model(batch['src_seq'].cuda(), batch['length'].cuda(), infer_mean=True)
            logits, mask = model(batch['src_seq'], batch['length'], infer_mean=True)
        sample_logits = logits[0, :length, :length, -1].detach()
        # triangle mask
        if mean_triual:
            low_tr = torch.tril(sample_logits, diagonal=-1)
            upp_tr = torch.triu(sample_logits, diagonal=1)
            mean_logits = (low_tr.t() + upp_tr) / 2
            sample_logits = mean_logits + mean_logits.t()
    
        pred_mat = torch.sigmoid(sample_logits) > 0.5
        pred = pred_mat.cpu().numpy()
    
        # convert preds to triplets of base pair positions as spotrna does (p1, p2, 0)
        pairs = [(i, j, 0) for i in range(pred.shape[0]) for j in range(pred.shape[1]) if pred[i, j]]
    
        return pairs


        # return pairs

# 
# def infer_rnaformer(model_dir, config_dir, sequence, gpu=-1):
    # TODO: add option if model does not yet exist to train from scratch



