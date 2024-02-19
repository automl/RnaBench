import torch
import pandas as pd
import numpy as np

from pathlib import Path
from collections import defaultdict

from RnaBench.lib.data.parser.pdb_parser import MMCIF3DParser

class Rna3dDataset():
    def __init__(
                 self,
                 data_dir: str,
                 transform = None,
                 padding_symbol = '<pad>',
                 device='cpu',
                 ):
        self.device=device
        self.parser = MMCIF3DParser(data_dir)
        self.df = self.parser.parse_dir2df()
        self.max_structure_length = self.df.groupby(['solo_accession', 'chain']).count()['x_coordinate'].max()  # self.df['atom_position'].max()
        print(self.max_structure_length)

        sequence_map = defaultdict(list)
        # TODO: Put the sequence mapping into an apply!!!
        for name, _ in self.df.groupby(['solo_accession', 'chain', 'nucleotide', 'sequence_position']):
            sequence_map[name[0]+'_'+name[1]].append([name[2], name[3]])
        sequence_map = [{'sequence': [nuc[0] for nuc in sorted(v, key=lambda x: x[1])],
                         'accession': '_'.join(k.split('_')[:-1]),
                         'chain': k.split('_')[-1]}
                         for k, v in sequence_map.items()]
        self.sequence_info = pd.DataFrame.from_dict(sequence_map)
        self.sequence_info.loc[:, 'length'] = self.sequence_info['sequence'].apply(len)
        self.max_sequence_length = self.sequence_info['length'].max()
        print(self.max_sequence_length)
        self.sequence_info.set_index(['accession', 'chain'], inplace=True)

        self.transform = transform
        self.padding_symbol = padding_symbol
        self.chain_vocab = [self.padding_symbol] + list(self.df['chain'].unique())
        self.nucleotide_vocab = [self.padding_symbol] + list(self.df['nucleotide'].unique())
        self.accession_vocab = [self.padding_symbol] + list(self.df['solo_accession'].unique())
        self.atom_type_vocab = [self.padding_symbol] + list(self.df['atom_type'].unique())
        self.atom_id_vocab = [self.padding_symbol] + list(self.df['atom_id'].unique())
        self.atom_symbol_vocab = [self.padding_symbol] + list(self.df['atom_symbol'].unique())

        # print(self.atom_id_vocab)
        self.accession_stoi = {s: i for i, s in enumerate(self.accession_vocab)}
        self.chain_stoi = {s: i for i, s in enumerate(self.chain_vocab)}
        self.nucleotide_stoi = {s: i for i, s in enumerate(self.nucleotide_vocab)}
        self.atom_type_stoi = {s: i for i, s in enumerate(self.atom_type_vocab)}
        self.atom_id_stoi = {s: i for i, s in enumerate(self.atom_id_vocab)}
        self.atom_symbol_stoi = {s: i for i, s in enumerate(self.atom_symbol_vocab)}

        # print(self.atom_id_stoi)
        self.accession_itos = {i: s for s, i in self.accession_stoi.items()}
        self.chain_itos = {i: s for s, i in self.chain_stoi.items()}
        self.nucleotide_itos = {i: s for s, i in self.nucleotide_stoi.items()}
        self.atom_type_itos = {i: s for s, i in self.atom_type_stoi.items()}
        self.atom_id_itos = {i: s for s, i in self.atom_id_stoi.items()}
        self.atom_symbol_itos = {i: s for s, i in self.atom_symbol_stoi.items()}


        self.data = self.to_rna_3d_structure()
        self.data = {k: v.to_numeric(nuc_stoi=self.nucleotide_stoi,
                                     atom_id_stoi=self.atom_id_stoi,
                                     atom_symbol_stoi=self.atom_symbol_stoi,
                                     atom_type_stoi=self.atom_type_stoi,
                                     chain_stoi=self.chain_stoi,
                                     accession_stoi=self.accession_stoi,
                                     ).apply_padding(
                                        max_length=self.max_structure_length,
                                        padding_with=self.atom_id_stoi[self.padding_symbol],
                                        max_seq_length=self.max_sequence_length,
                                     ) for k, v in self.data.items()}

    def to_rna_3d_structure(self):
        return {i: rna for i, rna in map(self.to_3d_rna, enumerate(self.df.groupby(['solo_accession', 'chain'])))}

    def to_3d_rna(self, data):
        idx = data[0]
        accession = data[1][0][0]
        chain = data[1][0][1]
        group = data[1][1]
        sequence = self.sequence_info.loc[(accession, chain)]['sequence']
        return idx, RnaSingleChain3dStructure(id=idx,
                                              accession=accession,
                                              chain=chain,
                                              sequence=sequence,
                                              data=group,
                                              )

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        # print(sample.to_numpy().to_dict()['x_coordinate'])
        # print(sample.to_torch(device='cpu'))
        sample = sample.to_torch(device=self.device)

        if self.transform:
            for trfm in self.transform:
                sample = trfm(sample)
        return sample.to_dict()



class RnaSingleChain3dStructure():
    def __init__(self,
                 id,
                 accession,
                 chain,
                 sequence,
                 data,
                 ):
        # self.data = data
        if isinstance(data, pd.DataFrame):
            self._id = [id]
            self._accession = [accession]
            self._chain = [chain]
            self._length = [len(data['x_coordinate'].to_list())]# [data['atom_position'].max()]
            self._sequence_length = [len(sequence)]
            self._x = data['x_coordinate'].to_list()
            self._y = data['y_coordinate'].to_list()
            self._z = data['z_coordinate'].to_list()
            self._nucleotides = data['nucleotide'].to_list()
            self._nucleotide_positions = data['sequence_position'].to_list()
            self._atom_positions = data['atom_position'].to_list()
            self._atom_symbols = data['atom_symbol'].to_list()
            self._atom_ids = data['atom_id'].to_list()
            self._atom_types = data['atom_type'].to_list()
            self._B_iso_or_equiv = data['B_iso_or_equiv'].to_list()
            self._occupancy = data['occupancy'].to_list()
            self._sequence = sequence
        elif isinstance(data, pd.Series) or isinstance(data, dict):
            self._id = data['Id']
            self._accession = data['Accession']
            self._chain = data['Chain']
            self._length = data['length']  # [len(data['x_coordinate'])]
            self._sequence_length = data['sequence_length']
            self._x = data['x_coordinate']
            self._y = data['y_coordinate']
            self._z = data['z_coordinate']
            self._nucleotides = data['nucleotide']
            self._nucleotide_positions = data['sequence_position']
            self._atom_positions = data['atom_position']
            self._atom_symbols = data['atom_symbol']
            self._atom_ids = data['atom_id']
            self._atom_types = data['atom_type']
            self._B_iso_or_equiv = data['B_iso_or_equiv']
            self._occupancy = data['occupancy']
            self._sequence = sequence
        else:
            raise UserWarning('RnaSingleChain3dStructure received unknown datatype', type(data))
        # print(self.id)
        # print(self.length)

    def __iter__(self):
        for att in vars(self):
            yield att

    def __getitem__(self, item):
        return getattr(self, item)

    def items(self):
        for att in self:
            yield att, self[att]

    def to_dict(self):
        return {
          'Id': self.id,
          'Accession': self.accession,
          'Chain': self.chain,
          'length': self.length,
          'sequence': self.sequence,
          'sequence_length': self.sequence_length,
          'x_coordinate': self.x,
          'y_coordinate': self.y,
          'z_coordinate': self.z,
          'nucleotide': self.nucleotides,
          'sequence_position': self.nucleotide_positions,
          'atom_position': self.atom_positions,
          'atom_symbol': self.atom_symbols,
          'atom_id': self.atom_ids,
          'atom_type': self.atom_types,
          'B_iso_or_equiv': self.B_iso_or_equiv,
          'occupancy': self.occupancy,
        }

    def to_numeric(self,
                   nuc_stoi,
                   atom_id_stoi,
                   atom_symbol_stoi,
                   atom_type_stoi,
                   chain_stoi,
                   accession_stoi,
                   ):
        d = self.to_dict()
        d['Chain'] = [chain_stoi[i] for i in d['Chain']]
        d['Accession'] = [accession_stoi[i] for i in d['Accession']]
        d['nucleotide'] = [nuc_stoi[i] for i in d['nucleotide']]
        d['atom_symbol'] = [atom_symbol_stoi[i] for i in d['atom_symbol']]
        d['atom_type'] = [atom_type_stoi[i] for i in d['atom_type']]
        d['atom_id'] = [atom_id_stoi[i] for i in d['atom_id']]
        d['sequence'] = [nuc_stoi[i] for i in d['sequence']]
        # data = pd.Series(d)
        return RnaSingleChain3dStructure(id=d['Id'][0],
                                         accession=d['Accession'][0],
                                         chain=d['Chain'][0],
                                         sequence=d['sequence'],
                                         data=d)



    def apply_padding(self,
                      max_length,
                      max_seq_length,
                      padding_with,
                      ):
        d = self.to_dict()
        # print(d)
        # print(max_length, d['length'][0])
        pad_len = max_length - d['length'][0]
        seq_pad_len = max_seq_length - d['sequence_length'][0]
        # print('pad', d['length'][0] + pad_len, pad_len)

        padding = [padding_with] * pad_len
        seq_padding = [padding_with] * seq_pad_len
        new_d = {}
        for k, v in d.items():
            if k in ['Id', 'Accession', 'Chain', 'length', 'sequence_length']:
                new_d[k] = v
            elif k == 'sequence':
                new_d[k] = v + seq_padding
            else:
                new_d[k] = v + padding
            # print(k, len(new_d[k]))  # , len(v), k, type(v))
        # print(new_d)
        data = pd.Series(new_d)
        return RnaSingleChain3dStructure(id=new_d['Id'][0],
                                         accession=new_d['Accession'][0],
                                         chain=new_d['Chain'][0],
                                         sequence=new_d['sequence'],
                                         data=new_d)

    def to_numpy(self):
        d = self.to_dict()
        d = {k: np.asarray(v) for k, v in d.items()}
        # data = pd.Series(d)
        return RnaSingleChain3dStructure(id=d['Id'][0],
                                         accession=d['Accession'][0],
                                         chain=d['Chain'][0],
                                         sequence=d['sequence'],
                                         data=d)

    def to_torch(self, device):
        if not isinstance(self.x, np.ndarray):
            new_rna = self.to_numpy()
        kwargs = {}
        # print(new_rna.to_dict())
        for att, value in new_rna.to_dict().items():
            # print(att, value)
            # print(torch.tensor(value))
            # print(torch.FloatTensor(value))
            # print(torch.LongTensor(value)[0])
            if device == 'cpu':
                if type(value[0]) == np.str_:
                    raise UserWarning("Torch task contains instance of unsupported type:", type(value[0]))
                elif type(value[0]) == np.float64:
                    if len(value) > 1:
                        kwargs[att] = torch.FloatTensor(value)
                    else:
                        kwargs[att] = torch.FloatTensor(value)[0]
                else:
                    if len(value) > 1:
                        kwargs[att] = torch.LongTensor(value)
                    else:
                        kwargs[att] = torch.LongTensor(value)[0]
            else:
                if type(value[0]) == np.str_:
                    raise UserWarning("Torch task contains instance of unsupported type:", type(value[0]))
                elif type(value[0]) == np.float64:
                    if len(value) > 1:
                        kwargs[att] = torch.cuda.FloatTensor(value, device=device)
                    else:
                        kwargs[att] = torch.cuda.FloatTensor(value, device=device)[0]
                else:
                    if len(value) > 1:
                        kwargs[att] = torch.cuda.LongTensor(value, device=device)
                    else:
                        kwargs[att] = torch.cuda.LongTensor(value, device=device)[0]
        # print(kwargs)
        # data = pd.Series(kwargs)
        # print(data)
        return RnaSingleChain3dStructure(
                                         id=kwargs['Id'],
                                         accession=kwargs['Accession'],
                                         chain=kwargs['Chain'],
                                         sequence=kwargs['sequence'],
                                         data=kwargs)





    @property
    def id(self):
        return self._id

    @property
    def accession(self):
        return self._accession

    @property
    def chain(self):
        return self._chain

    @property
    def length(self):
        return self._length

    @property
    def sequence(self):
        return self._sequence

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def nucleotides(self):
        return self._nucleotides

    @property
    def nucleotide_positions(self):
        return self._nucleotide_positions

    @property
    def atom_ids(self):
        return self._atom_ids

    @property
    def atom_positions(self):
        return self._atom_positions

    @property
    def atom_symbols(self):
        return self._atom_symbols

    @property
    def atom_types(self):
        return self._atom_types

    @property
    def B_iso_or_equiv(self):
        return self._B_iso_or_equiv

    @property
    def occupancy(self):
        return self._occupancy



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from Bio.SVDSuperimposer import SVDSuperimposer

    # mmcif_dir = "data/3D/solo_member_cif_1_5__3_286"
    mmcif_dir = "data/3D/solo_representative_cif_1_5__3_286"
    # mmcif_dir = "data/3D/solo_member_cif_all__3_284"
    rna_3d_dataset = Rna3dDataset(mmcif_dir, device=0)
    sup = SVDSuperimposer()
    data_iterator = DataLoader(rna_3d_dataset, batch_size=64)

    for i_batch, sampled_batch in enumerate(data_iterator):
        # print(sampled_batch['x_coordinate'].size())
        for b, length in enumerate(sampled_batch["sequence_length"].detach().cpu().numpy()):
            print([rna_3d_dataset.nucleotide_itos[i] for i in sampled_batch['sequence'][b, :length].detach().cpu().numpy()])
        for b, length in enumerate(sampled_batch["length"].detach().cpu().numpy()):
            x = sampled_batch['x_coordinate'][b, :length].detach().cpu().numpy()
            y = sampled_batch['y_coordinate'][b, :length].detach().cpu().numpy()
            z = sampled_batch['z_coordinate'][b, :length].detach().cpu().numpy()
            coords1 = np.stack([x, y, z], axis=1)
            coords2 = np.stack([x, y, z], axis=1)

            sup.set(coords1, coords2)
            sup.run()
            rmsd = sup.get_rms()
            print('RMSD1:', np.round(rmsd, 4))

            coords2 = np.stack([y, x, z], axis=1)
            sup.set(coords1, coords2)
            sup.run()
            rmsd = sup.get_rms()
            print('RMSD2:', np.round(rmsd, 4))


