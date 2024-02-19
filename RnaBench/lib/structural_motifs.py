import numpy as np


class Stem:
    """
    RNA stem.
    Provides attributes of an RNA stem.
    """

    def __init__(
            self,
            rna_id,
            stem_id,
            length=None,
            rng_5p=None,
            rng_3p=None,
            seq=None
    ):
        """
        TODO
        """
        self._rna_id = rna_id
        self._stem_id = stem_id
        self._length = length
        self._rng_5p = rng_5p
        self._rng_3p = rng_3p
        self._seq = seq

        if self._rng_3p is not None and self._length is None:
            self._length = self._rng_3p[1] - self._rng_3p[0]

    def __len__(self):
        return self._length

    def from_bprna_list(self, info_list):
        self._rng_5p = np.array(info_list[0].split("..")).astype(int)
        self._rng_3p = np.array(info_list[2].split("..")).astype(int)
        self._seq = [info_list[1].strip('"'), info_list[3].strip('"')]
        self._length = len(self._seq[0])

    @property
    def stem_id(self):
        return self._stem_id

    @property
    def rna_id(self):
        return self._rna_id


class ILoop:
    """
    RNA internal loop.
    Provides attributes of an RNA internal loop.
    """

    def __init__(
            self,
            rna_id,
            iloop_id,
            size=None,
            cp_5p=None,
            cp_5p_nuc=None,
            cp_3p=None,
            cp_3p_nuc=None,
            seq=None
    ):
        """
        TODO
        :param rna_id: str
        :param iloop_id: str
        :param size: (int, int)
        :param cp_5p: (int, int)
        :param cp_5p_nuc:  (str, str)
        :param cp_3p: (int, int)
        :param cp_3p_nuc: (str, str)
        :param seq:  (str, str)

        """
        self._rna_id = rna_id
        self._iloop_id = iloop_id
        self._size = size
        self._cp_5p = cp_5p
        self._cp_3p = cp_3p
        self._cp_5p_nuc = cp_5p_nuc
        self._cp_3p_nuc = cp_3p_nuc
        self._seq = seq

        if self._size is None and self._seq is not None:
            self._size = (len(self._seq[0]), len(self._seq[1]))


    def size(self, dim=None):
        size_req = self._size if dim is None else self._size[dim]
        return size_req

    def from_bprna_list(self, info_list):

        self._seq = [info_list[0][1].strip('"'), info_list[1][1].strip('"')]
        self._cp_5p = np.array(info_list[0][2].strip('()').split(",")).astype(int)
        self._cp_3p = np.array(info_list[1][2].strip('()').split(",")).astype(int)

        self._cp_5p_nuc = info_list[0][3].split(":")
        self._cp_3p_nuc = info_list[1][3].split(":")
        self._size = (len(self._seq[0]), len(self._seq[1]))

    @property
    def iloop_id(self):
        return self._iloop_id

    @property
    def rna_id(self):
        return self._rna_id


class Hairpin:
    """
    RNA hairpin.
    Provides attributes of an RNA stem.
    """

    def __init__(
            self,
            rna_id,
            hp_id,
            length=None,
            rng=None,
            seq=None,
            cp=None
    ):
        """
        TODO
        """
        self._rna_id = rna_id
        self._hp_id = hp_id
        self._length = length
        self._rng = rng
        self._seq = seq
        self._cp = cp
        if self._rng is not None and self._length is None:
            self._length = len(self._seq)

    def __len__(self):
        return self._length

    def from_bprna_list(self, info_list):
        # H1 20..26 "GAUAUGG" (19,27) G:C PK{3,4}
        self._rng = np.array(info_list[0].split("..")).astype(int)

        self._length = self._rng[1]-self._rng[0]+1
        self._seq = info_list[1].strip('"')
        self._cp = info_list[3].split(":")

        # self._rng_5p = np.array(info_list[0].split("..")).astype(int)
        # self._rng_3p = np.array(info_list[2].split("..")).astype(int)
        # self._seq = [info_list[1].strip('"'), info_list[3].strip('"')]
        # self._length = len(self._seq[0])

    @property
    def hp_id(self):
        return self._hp_id

    @property
    def rna_id(self):
        return self._rna_id


class MLoop:
    """
    RNA internal loop.
    Provides attributes of an RNA internal loop.
    """

    def __init__(
            self,
            rna_id,
            mloop_id,
            size=None,
            nuc_cps= None,
            cps=None,
            seqs=None
    ):
        """
        TODO
        :param rna_id: str
        :param iloop_id: str
        :param size: tuple(int)
        :param cps: list(tuple(int, int))
        :param nuc_cps: list(tuple(str, str))
        :param seqs:  list(str)

        """
        self._rna_id = rna_id
        self._mloop_id = mloop_id
        self._size = size
        self._cps = cps
        self._nuc_cps = nuc_cps
        self._seqs = seqs

        if self._size is None and self._seqs is not None:
            self._size = tuple([len(seq) for seq in self._seqs])

        if self._size is None and self._cps is not None:
            size_list = []
            for i in range(len(self._cps)):
                size_list.append()

        #if self._size is None and self._cps is not None:
        #    self._size = (cp[1] - cp[0] + 1 if cp is not None else 0 for cp in self._cps)

    def size(self, dim=None):
        size_req = self._size if dim is None else self._size[dim]
        return size_req

    def from_bprna_list(self, info_list):

        self._seqs = [strand[1].strip('"') for strand in info_list]
        self._cps = [np.array(strand[2].strip('()').split(",")).astype(int) for strand in info_list]
        self._nuc_cps = [np.array(strand[3].split(":")) for strand in info_list]
        self._size = tuple([len(self._seqs) for seq in self._seqs])

    @property
    def iloop_id(self):
        return self._iloop_id

    @property
    def rna_id(self):
        return self._rna_id

