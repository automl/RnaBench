import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

from RnaBench.lib.rna_folding_algorithms.DL.ufold.Network import U_Net as FCNNet

from RnaBench.lib.rna_folding_algorithms.DL.ufold.utils import *
from RnaBench.lib.rna_folding_algorithms.DL.ufold.config import process_config
import pdb
import time
from RnaBench.lib.rna_folding_algorithms.DL.ufold.data_generator import RNASSDataGenerator, Dataset,RNASSDataGenerator_input
from RnaBench.lib.rna_folding_algorithms.DL.ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from RnaBench.lib.rna_folding_algorithms.DL.ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections

import subprocess

from RnaBench.lib.rna_folding_algorithms.DL.ufold.postprocess import postprocess_new_nc as postprocess_nc
from RnaBench.lib.rna_folding_algorithms.DL.ufold.postprocess import postprocess_new as postprocess


RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
rng = np.random.default_rng(seed=0)
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
    'N': rng.choice(['A', 'C', 'G', 'U']),  # 'N',
    'A': 'A',
    'U': 'U',
    'C': 'C',
    'G': 'G',
}

class UFold():
    def __init__(self,
                 model='ufold_train.pt',  # available models: ufold_train.pt, ufold_train_alldata.pt, ufold_train_pdbfinetune.pt
                 device = 'cpu',
                 nc = True,
                 ):
        self.nc = nc

        torch.multiprocessing.set_sharing_strategy('file_system')
        config_file = 'RnaBench/lib/rna_folding_algorithms/DL/ufold/config.json'
        config = process_config(config_file)
    
        d = config.u_net_d
        self.BATCH_SIZE = config.batch_size_stage_1
        OUT_STEP = config.OUT_STEP
        LOAD_MODEL = config.LOAD_MODEL
        data_type = config.data_type
        model_type = config.model_type
        epoches_first = config.epoches_first
        self.model_name = model
    
        self.MODEL_SAVED = f'RnaBench/lib/rna_folding_algorithms/DL/ufold/models/{self.model_name}'
    
        self.device = device
    
        seed_torch()

    def __name__(self):
        return 'UFold_' + self.model_name + '_' + f'nc_{self.nc}'
    
    def __repr__(self):
        return 'UFold_' + self.model_name + '_' + f'nc_{self.nc}'
    
        
    def __call__(self, sequence):
        sequence = [NUCS[i] for i in sequence]

        test_data = RNASSDataGenerator_input(sequence)  # input sequence as list
    
        params = {'batch_size': self.BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': 1,
                  'drop_last': False}
    
        test_set = Dataset_FCN(test_data)
        test_generator = data.DataLoader(test_set, **params)
        contact_net = FCNNet(img_ch=17)
    
        contact_net.load_state_dict(torch.load(self.MODEL_SAVED,map_location=self.device))
        contact_net.to(self.device)

        pairs = self.model_eval_all_test(contact_net,test_generator)

        return pairs

    def model_eval_all_test(self, contact_net,test_generator):
        contact_net.train()
        result_no_train = list()
        result_no_train_shift = list()
        seq_lens_list = list()
        batch_n = 0
        seq_names = []
        ct_dict_all = {}
        dot_file_dict = {}
        pos_weight = torch.Tensor([300]).to(self.device)
        criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
            pos_weight = pos_weight)
        for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
            batch_n += 1
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(self.device)
            seq_ori = torch.Tensor(seq_ori.float()).to(self.device)

            with torch.no_grad():
                pred_contacts = contact_net(seq_embedding_batch)
    
            if self.nc:
                u_no_train = postprocess_nc(pred_contacts,
                    seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            else:
                u_no_train = postprocess(pred_contacts,
                    seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            map_no_train = (u_no_train > 0.5).float()

            threshold = 0.5
            th = 0

            if seq_name[0].startswith('.'):
                seq_name = [seq_name[0][1:]]
            seq_names.append(seq_name[0].replace('/','_'))

            ct_dict_all = get_ct_dict_fast(map_no_train,batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])  # ,dot_file_dict,tertiary_bp

            return [(i, j, 0) for i, j in sorted(set(ct_dict_all[1]), key=lambda x: x[0])]


def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmpp = np.copy(seq_tmp)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    letter='AUCG'
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])

    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    ct_dict[batch_num] = [tuple(sorted((max(0, seq[0][i]-1),max(0, seq[1][i]-1)))) for i in np.arange(len(seq[0])) if seq[0][i] != 0]  # add -1 for zero indexing
    return ct_dict


