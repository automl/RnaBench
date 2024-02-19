import numpy as np
import torch

from typing import Optional
from RnaBench.lib.alphabets import SpecialSymbols

class TaskFactory():
    def __init__(
            self,
            dataset,
            seed: int = 0,
            task: str = 'inverse_rna_folding',
            numpy: bool = False,
            padding_character: str = SpecialSymbols.pad,
            unknown_character: str = SpecialSymbols.unknown,
            bos_character: str = SpecialSymbols.bos,
            eos_character: str = SpecialSymbols.eos,
            replacement_factor: float = 0.2,
            num_max_replacements: int = 5,
            padding: Optional[str] = None,
    ):
        """
        Turns samples in a dataset to RNA design tasks.
        """
        self.rng = np.random.default_rng(seed=seed)
        self.task = task
        self.dataset = dataset
        self.max_length = dataset.max_seq_length
        self.max_pair_length = dataset.max_pair_length
        self.numpy = numpy
        self.seq_stoi = dataset.seq_stoi
        self.struc_stoi = dataset.struc_stoi

        # print(self.seq_stoi)
        # print(self.struc_stoi)

        self.padding = padding
        self.unknown_character = unknown_character
        self.bos = bos_character
        self.eos = eos_character
        self.replacement_factor = replacement_factor
        self.num_max_replacements = num_max_replacements
        if padding:
            self.padding_character = padding_character
            if not self.padding_character in self.seq_stoi:
                padding_index = max(len(self.dataset.structure_vocab), len(self.dataset.sequence_vocab))
                self.seq_stoi.update({self.padding_character: padding_index})
            if not self.padding_character in self.struc_stoi:
                padding_index = max(len(self.dataset.structure_vocab), len(self.dataset.sequence_vocab))
                self.struc_stoi.update({self.padding_character: padding_index})

        self.unknown_index = SpecialSymbols.unknown_idx

        self.seq_stoi.update({self.unknown_character: self.unknown_index})
        self.struc_stoi.update({self.unknown_character: self.unknown_index})

        self.seq_itos = {i: s for s, i in self.seq_stoi.items()}
        self.struc_itos = {i: s for s, i in self.struc_stoi.items()}

        # self.prepare_tasks()

    def prepare_tasks(self):
        self.tasks = list(map(self.get_task, self.dataset))

    def get_task(self, rna):
        """
        Create a task from the data.
        """
        # task_seq = self.get_task_seq(rna)
        # print(type(rna._sequence.length))
        pairs = rna.pairs
        length = len(rna) if not self.numpy else np.asarray([len(rna)], dtype=np.int32)
        num_pairs = rna.num_pairs if not self.numpy else np.asarray([rna.num_pairs], dtype=np.int32)
        if self.padding:
            pad_seq = self.apply_padding(rna.sequence)
            pairs = self.apply_pair_padding(rna.pairs)
            # pad_struc = self.apply_padding(rna.structure)
            rna.set_sequence(pad_seq)
            rna.set_structure()  # structure has now length of sequence
            rna.prepare_matrix(length=len(pad_seq))  # and matrix, too
        if self.numpy:
            rna = rna.to_numpy(self.seq_stoi, self.struc_stoi)
            pairs = np.asarray([p if isinstance(p[0], int) else [self.struc_stoi[i] for i in p] for p in pairs])
            # print(self.seq_stoi)
            # print(task_seq)
            # task_seq = np.asarray([self.seq_stoi[s] for s in task_seq], dtype=np.int32)
            # task_struc = np.asarray([self.struc_stoi[s] for s in task_struc], dtype=np.int32)
        # task_id = rna.id
        # target_seq = rna.sequence
        # target_pairs = pairs
        # target_mat = rna.matrix
        # gc_content = rna.gc
        # energy = rna.energy
        # print(rna.matrix.shape)
        # length = len(rna) if not self.numpy else np.asarray([len(rna)], dtype=np.int32)


        return RnaDesignTask(
            task_id=rna.id,
            sequence=rna.sequence,
            # task_sequence=rna.sequence,
            # task_struc=task_struc,
            # task_pairs=target_pairs,
            pairs=pairs,
            # target_struc=target_struc,
            gc_content=rna.gc,
            # energy=energy,
            matrix=rna.matrix,
            length=length,
            num_pairs=num_pairs,
        )

    def get_task_seq(self, rna):
        if self.task == 'inverse_rna_folding':
            task_seq = [self.unknown_character] * len(rna.sequence)
        elif self.task == 'constrained_inverse_folding':
            task_seq = self.random_mask(rna.sequence)
        else:
            task_seq = rna.sequence

        return task_seq

    def random_mask(self, s):
        """
        Replace random regions of string with unknown character.
        Not simple random masking, we want the replacements not to be
        position-wise but region-wise.
        """
        rep_max_len = int(len(s) * self.replacement_factor)
        # s_list = list(s)
        for i in range(self.num_max_replacements):
            rep_length = self.rng.integers(0, rep_max_len)
            rep_start_pos = self.rng.integers(0, len(s) - rep_length + 1)
            s_list[rep_start_pos:rep_start_pos + rep_length] = [self.unknown_character] * rep_length
        return s_list

    def apply_padding(self, s):
        # print('padding', s)
        if self.padding == 'end':
            if self.bos and self.eos:
                return [self.bos] + s + [self.eos] + [self.padding_character] * (self.max_length - len(s))
            else:
                return s + [self.padding_character] * (self.max_length - len(s))
        elif self.padding == 'start':
            if self.bos and self.eos:
                return [self.padding_character] * (self.max_length - len(s)) + [self.bos] + s + [self.eos]
            else:
                return [self.padding_character] * (self.max_length - len(s)) + s
        elif self.padding == 'equal':
            pad_length = int((self.max_length - len(s)) / 2)
            if self.bos and self.eos:
                return [self.padding_character] * pad_length + [self.bos] + s + [self.eos] + [self.padding_character] * pad_length
            else:
                return [self.padding_character] * pad_length + s + [self.padding_character] * pad_length

    def apply_pair_padding(self, p):
        if self.padding == 'end':
            return p + [[self.padding_character, self.padding_character, self.padding_character]] * (self.max_pair_length - len(p))
        elif self.padding == 'start':
            return [[self.padding_character,self.padding_character, self.padding_character]] * (self.max_pair_length - len(p)) + p
        elif self.padding == 'equal':
            pad_length = int((self.max_pair_length - len(p)) / 2)
            return [[self.padding_character, self.padding_character, self.padding_character]] * pad_length + p + [self.padding_character, self.padding_character, self.padding_character] * pad_length




    def __iter__(self):
        for task in self.tasks:
            yield task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.get_task(self.dataset[i])
        # return self.tasks[i]


class BaseTask():
    """
    Provides basic functionality of a Task.
    Takes keyword arguments and sets attributes.
    Provides functionality for using a task as a dictionary.
    """

    def __init__(self, **kwargs):
        for k, arg in kwargs.items():
            setattr(self, k, arg)

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self):
        pass

    def __iter__(self):
        for att in vars(self):
            yield att

    def __repr__(self):
        return ', '.join([att for att in self])

    def items(self):
        for att in self:
            yield att, self[att]

    def keys(self):
        for att in self:
            yield att

    def values(self):
        for att in self:
            yield self[i]

    def to_torch(self):
        pass

    def to_numpy(self):
        pass

    def to_numeric(self):
        pass

    def to_dict(self):
        return {att: self[att] for att in self}


class RnaDesignTask(BaseTask):
    """
    class to produce a RNA Design Task.
    A Task currently requires:
        task_id: ID of the task
        task_seq (str): the sequence of the task
        task_struc (str): the structure of the task
        target_seq (str): the ground truth sequence of the RNA
        target_struc (str): the ground truth structure of the RNA
    Also provides functionality to create numpy arrays or torch Tensors from the
    Inputs.
    """

    def __init__(self, **kwargs):
        """
        TODO
        """
        super().__init__(**kwargs)
        assert hasattr(self, "task_id"), "RnaDesignTask has no ID"
        assert hasattr(self, "sequence"), "RnaDesignTask has no sequence"
        assert hasattr(self, "pairs"), "RnaDesignTask has no pairs"
        # assert hasattr(self, "task_struc"), "RnaDesignTask has no task structure"
        # assert hasattr(self, "target_struc"), "RnaDesignTask has no ground truth structure"
        # assert hasattr(self, "target_sequence"), "RnaDesignTask has no ground truth sequence"

    def to_numpy(self):
        """
        Translate all attributes to numpy arrays.
        """
        kwargs = {}
        for att, value in self.items():
            if isinstance(value, str):
                kwargs[att] = np.asarray(list(value), dtype=np.str_)
            elif isinstance(value, float):
                kwargs[att] = np.asarray([value], dtype=np.float64)
            elif isinstance(value, int):
                kwargs[att] = np.asarray([value], dtype=np.int32)
            else:
                kwargs[att] = np.asarray(value, dtype=np.int32)
        return RnaDesignTask(**kwargs)

    def to_torch(self, device):
        """
        Translate all attributes to torch Tensors.
        """
        # print(gc.garbage)
        # new_task = self.to_numpy()
        kwargs = {}
        for att, value in self.items():
            # print(att, value)
            if device == 'cpu':
                if value.dtype == np.str_:
                    raise UserWarning("Torch task contains instance of unsupported type:", value.dtype)
                elif value.dtype == np.float64:
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
                if value.dtype == np.str_:
                    raise UserWarning("Torch task contains instance of unsupported type:", value.dtype)
                elif value.dtype == np.float64:
                    if len(value) > 1:
                        kwargs[att] = torch.cuda.FloatTensor(value, device=device)
                    else:
                        kwargs[att] = torch.cuda.FloatTensor(value, device=device)[0]
                else:
                    if len(value) > 1:
                        kwargs[att] = torch.cuda.LongTensor(value, device=device)
                    else:
                        kwargs[att] = torch.cuda.LongTensor(value, device=device)[0]
        return RnaDesignTask(**kwargs)

    def __len__(self):
        return len(self.target_sequence)

    @property
    def id(self):
        return self.task_id


class RnaSequence():
    """
    A RNA sequence object.
    Provides attributes of a RNA sequence.
    """

    def __init__(self, seq_id, sequence, gc=None, length=None):
        """
        TODO
        """
        self._id = seq_id
        self._sequence = sequence
        # print(self._sequence)
        if gc is not None:
            self._gc = gc
        else:
            # print(sequence)
            self._gc = (''.join(sequence).upper().count('G') + ''.join(sequence).upper().count('C')) / len(
                ''.join(sequence))
        if length is not None:
            self.length = length
        else:
            self.length = len(sequence)

    def __len__(self):
        if isinstance(self.length, int) or isinstance(self.length, np.int64):
            return self.length
        else:
            return self.length[0]

    def __iter__(self):
        for s in self._sequence:
            yield s

    def to_numeric(self, stoi):
        """
        Change representaiton of sequence to integer.

        Input:
          stoi (dict): string to integer translation dict for sequence.
        Returns:
          New RnaSequence object.
        """
        new_seq = self.to_list()
        seq_id = new_seq.id
        seq = [stoi[s] for s in new_seq.sequence]
        gc = new_seq.gc
        return RnaSequence(seq_id=seq_id, sequence=seq, gc=gc, length=new_seq.length)

    def to_numpy(self, stoi):
        """
        Translate attributes to numpy arrays.
        """
        new_seq = self.to_numeric(stoi)
        seq_id = np.asarray(new_seq.id, dtype=np.int32)
        seq = np.asarray(new_seq.sequence, dtype=np.int32)
        gc = np.asarray(new_seq.gc, dtype=np.float64)
        length = np.asarray(new_seq.length, dtype=np.int32)
        return RnaSequence(seq_id=seq_id, sequence=seq, gc=gc, length=length)

    def to_list(self):
        """
        Translate sequence to list representation.
        """
        seq_id = [self._id]
        seq = self._sequence
        gc = [self._gc]
        length = [self.length]
        return RnaSequence(seq_id=seq_id, sequence=seq, gc=gc, length=length)

    @property
    def sequence(self):
        return self._sequence

    @property
    def gc(self):
        return self._gc

    @property
    def id(self):
        return self._id


class RnaStructure():
    """
    RNA structure object.
    Provides attributes of a RNA structure.
    """

    def __init__(
            self,
            struc_id,
            # structure,
            # pos1id,
            # pos2id,
            # pk,
            pairs,
            # seq_length=None,
            # energy=None,
            matrix=False,
            length=None,
    ):
        """
        TODO
        """
        self._id = struc_id
        # self._structure = structure
        # self.pos1id = pos1id
        # self.pos2id = pos2id
        # self.pk = pk
        self.pairs = pairs
        # if energy is None:
        #     self._energy = np.random.randint(-100, 0)
        # else:
        # self._energy = energy

        if length is None:
            self.length = len(structure)
        else:
            self.length = length

        if isinstance(matrix, bool) and matrix:
            self.prepare_matrix(length=length)
        else:
            self.matrix = matrix

    def __len__(self):
        if isinstance(self.length, list):
            return self.length[0]
        return self.length

    def __iter__(self):
        for p in self.pairs:
            yield pair
        # for s in self._structure:
        #     yield s

    # def iter_pairs(self):
    #     for p1, p2 in zip(self.pos1id, self.pos2id):
    #         yield (p1, p2)

    def prepare_matrix(self, length):
        """
        Prepare binary matrix representation from provided list of pairs.
        """
        self.matrix = np.zeros((length, length), dtype=np.int32)
        # list(map(self.pair_to_matrix, zip(self.pos1id, self.pos2id)))
        list(map(self.pair_to_matrix, self.pairs))

    def pair_to_matrix(self, pair):
        """
        Set one pair in matrix.
        """
        self.matrix[pair[0], pair[1]] = 1
        self.matrix[pair[1], pair[0]] = 1

    def to_numeric(self, stoi):
        """
        Translate structure to integer representation.

        Input:
          stoi (dict): string to integer translation dictionary for structure.

        Returns:
          new RnaStructure object.
        """
        new_struc = self.to_list()
        struc_id = new_struc.id
        # struc = [stoi[s] for s in new_struc.structure]
        # p1 = new_struc.pos1id
        # p2 = new_struc.pos2id
        # pk = new_struc.pk
        # energy = new_struc.energy
        pairs = self.pairs
        length = new_struc.length
        mat = new_struc.matrix
        return RnaStructure(
            struc_id=struc_id,
            pairs=pairs,
            # structure=struc,
            # pos1id=p1,
            # pos2id=p2,
            # pk=pk,
            # energy=energy,
            length=length,
            matrix=mat,
        )

    def to_numpy(self, stoi):
        """
        Translate all attributes to numpy arrays
        """
        new_struc = self.to_numeric(stoi)
        struc_id = np.asarray(new_struc._id, dtype=np.int32)
        # struc = np.asarray(new_struc._structure, dtype=np.int32)
        # p1 = np.asarray(new_struc.pos1id, dtype=np.int32)
        # p2 = np.asarray(new_struc.pos2id, dtype=np.int32)
        # pk = np.asarray(new_struc.pk, dtype=np.int32)
        # energy = np.asarray(new_struc._energy, dtype=np.float64)
        pairs = np.asarray(self.pairs)
        length = np.asarray(new_struc.length, dtype=np.int32)
        mat = new_struc.matrix
        return RnaStructure(
            struc_id=struc_id,
            pairs=pairs,
            # structure=struc,
            # pos1id=p1,
            # pos2id=p2,
            # pk=pk,
            # energy=energy,
            matrix=mat,
            length=length,
        )

    def to_list(self):
        """
        Translate all attributes to list representation.
        """
        struc_id = [self._id]
        pairs = self.pairs
        # struc = self._structure
        # p1 = self.pos1id
        # p2 = self.pos2id
        # pk = self.pk
        # energy = [self._energy]
        length = [self.length]
        mat = self.matrix
        return RnaStructure(
            struc_id=struc_id,
            pairs=pairs,
            # structure=struc,
            # pos1id=p1,
            # pos2id=p2,
            # pk=pk,
            # energy=energy,
            length=length,
            matrix=mat,
        )

    # def set_energy(self, energy):
    #     self._energy = energy

    @property
    def num_pairs(self):
        return len(self.pairs)

    # @property
    # def energy(self):
    #     return self._energy

    @property
    def structure(self):
        return pairs2db(self.pairs, self.length)

    @property
    def id(self):
        return self._id


class RNA():
    """
    Provides RNA objects.
    Contains a RnaSequence and RnaStructure object.
    Provides attributes of a RNA.
    """

    def __init__(self,
                 rna_id,
                 sequence,
                 structure=None,
                 pairs=None,
                 gc=None,
                 # pos1id=None,
                 # pos2id=None,
                 # pk=None,
                 # energy=None,
                 matrix=None,
                 length=None,
                 **kwargs,
                 ):
        if not isinstance(sequence, RnaSequence):
            self._sequence = RnaSequence(seq_id=rna_id, sequence=sequence, length=length, gc=gc)
        else:
            self._sequence = sequence
        if not isinstance(structure, RnaStructure):
            # assert pos1id is not None, f"No pairs pos1id provided for RNA {rna_id}"
            # assert pos2id is not None, f"No pairs pos2id provided for RNA {rna_id}"
            # assert pk is not None, f"No PK information provided for RNA {rna_id}"
            self._structure = RnaStructure(
                struc_id=rna_id,
                pairs=pairs,
                # seq_length=len(sequence),
                # pos1id=pos1id,
                # pos2id=pos2id,
                # pk=pk,
                matrix=matrix,
                length=length
                # energy=energy,
            )
        else:
            self._structure = structure
        # Follwoign energy part should not be necessary but I'll keep it just in case
        # if self._structure.energy is None:
        #     rna_fold = RNAFoldAll(self._sequence.sequence)
        #     self._structure.set_energy = rna_fold.energy
        self._id = rna_id
        for k, arg in kwargs.items():
            setattr(self, k, arg)

    # def prepare_matrix(self):
    #     if self._structure.matrix is not None:
    #         self._structure.prepare_matrix(self.length)

    def __len__(self):
        return len(self._sequence)

    def to_list(self):
        new_sequence = self._sequence.to_list()
        new_structure = self.structure.to_list()
        return RNA(
            rna_id=new_sequence.id,
            sequence=new_sequence,
            structure=new_structure
        )

    def to_numpy(self, seq_stoi, struc_stoi):
        new_sequence = self._sequence.to_numpy(seq_stoi)
        new_structure = self._structure.to_numpy(struc_stoi)
        return RNA(
            rna_id=new_sequence.id,
            sequence=new_sequence,
            structure=new_structure
        )

    def to_numeric(self, seq_stoi, struc_stoi):
        new_sequence = self._sequence.to_numeric(seq_stoi)
        new_structure = self._structure.to_numeric(struc_stoi)
        return RNA(
            rna_id=new_sequence.id,
            sequence=new_sequence,
            structure=new_structure
        )

    def to_dict(self):
        return {
          'id': self.id,
          'sequence': self.sequence,
          'pairs': self.structure.pairs,
          'gc_content': self.gc,
          'matrix': self._structure.matrix,
          'length': len(self),
        }

    def set_sequence(self, seq):
        self._sequence = RnaSequence(seq_id=self.id, sequence=seq, length=len(seq))

    def set_structure(self):
        self._structure = RnaStructure(
                                       struc_id=self._structure.id,
                                       # structure=struc,
                                       pairs=self._structure.pairs,
                                       # seq_length=len(self),
                                       # pos1id=self._structure.pos1id,
                                       # pos2id=self._structure.pos2id,
                                       # pk=self._structure.pk,
                                       matrix=self._structure.matrix,
                                       length=len(self),
                                      )


    def prepare_matrix(self, length):
        if self._structure.matrix is not None:
            self._structure.prepare_matrix(length=length)


    @property
    def id(self):
        return self._id

    @property
    def sequence(self):
        return self._sequence.sequence

    @property
    def length(self):
        return len(self)

    # @property
    # def structure(self):
    #     return self._structure.structure

    @property
    def gc(self):
        return self._sequence.gc

    @property
    def num_pairs(self):
        return self._structure.num_pairs

    @property
    def matrix(self):
        return self._structure.matrix

    @property
    def pairs(self):
        return self._structure.pairs



    # @property
    # def structure_length(self):
    #     return len(self._structure)
