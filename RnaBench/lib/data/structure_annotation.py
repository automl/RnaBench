import subprocess
import json
import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
from collections import defaultdict, Counter


class DSSR():
    def __init__(self, dssr_dir):
        self.cwd = str(Path(dssr_dir).resolve())
        self.structure_infos = {}

    def run(self, mmcif_path):
        mmcif_path = str(Path(mmcif_path).resolve())
        args = [
            "./x3dna-dssr",
            f"-i={mmcif_path}",
            # "--symm",
            "--json",
        ]
        dssr = subprocess.Popen(args, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.json_str = dssr.communicate()[0].decode("utf-8")
        # self.err = dssr.communicate()[1].decode("utf-8")

    def json2dict(self):
        return json.loads(self.json_str)

    def parse_json(self):
        current_run_dict = {}
        run_infos = self.json2dict()
        # Fields to parse:
        # index = nucleotide index in all_chains
        # index_chain = index of nuc in chain
        # chain_name = chain
        # nt_resnum = original number of nuc in chain (as in provided pairs)
        # nt_name = name of nucleotide (might be modified, e.g. PSU)
        # nt_code = nucleotide code PSU -> P
        # is_modified = bool if nuc is modified
        # nt_id = nucleotide ID as in provided base pairs
        # nt_type = RNA, DNA, ...
        # dbn = dot-bracket notation for this position
        try:
            nucleotide_dict_list = run_infos['nts']
        except KeyError as e:
            raise UserWarning('DSSR: No nucleotides in output')
        nucleotides = [{'index': int(d['index'])-1,
                        'index_chain': int(d['index_chain'])-1,
                        'chain': d['chain_name'],
                        'nt_resnum': d['nt_resnum'],
                        'nt_name': d['nt_name'],
                        'nt_code': d['nt_code'],
                        'ID': d['nt_id']}
                        for d in nucleotide_dict_list]
        current_run_dict['sequences'] = self.get_chain_sequences(nucleotides)

        nuc_id2index_map = {d['ID']: (d['index'], d['index_chain'], d['chain']) for d in nucleotides}

        try:
            pairs_dict_list = run_infos['pairs']
        except KeyError as e:
            print('DSSR: Captured KeyError', e, '--> No Pairs in File')
            pairs_dict_list = []
        if pairs_dict_list:
            pairs = [{'nt1': d['nt1'],
                      'nt2': d['nt2'],
                      'name': d['name']}
                      for d in pairs_dict_list]
        else:
            pairs = []

        current_run_dict['pairs'] = self.resolve_pair_information(nuc_id2index_map, pairs)

        return current_run_dict

    def get_chain_sequences(self, nucleotides):
        chain_seqs = defaultdict(list)
        for d in nucleotides:
            chain_seqs['all'].insert(d['index'], d['nt_code'])
            chain_seqs[d['chain']].insert(d['index_chain'], d['nt_code'])
        return chain_seqs

    def resolve_pair_information(self, nuc_id2index_map, pairs):
        chain_pairs = defaultdict(list)
        for pair in pairs:
            chain_pairs['all'].append((nuc_id2index_map[pair['nt1']][0], nuc_id2index_map[pair['nt2']][0]))
            if nuc_id2index_map[pair['nt1']][2] == nuc_id2index_map[pair['nt2']][2]:  # same chain; intra chain pair
                chain_pairs[nuc_id2index_map[pair['nt1']][2]].append((nuc_id2index_map[pair['nt1']][1], nuc_id2index_map[pair['nt2']][1]))
        return chain_pairs


class BpRNA():
    def __init__(self, bprna_dir, working_dir):
        self.cwd = str(Path(bprna_dir).resolve())
        self.working_dir = str(Path(working_dir).resolve())
        self.bpseq_path = Path(self.working_dir, 'tmp.bpseq')
        self.st_file = Path(bprna_dir, 'tmp' + '.st')

    def run(self):
        p = subprocess.call(["perl", "bpRNA.pl", Path(self.bpseq_path.resolve())], cwd=self.cwd)  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(p)
        return p
        # stdout, stderr = p.communicate()


    def remove_multiplets(self, pos1, pos2):
        pos_count = Counter(pos1 + pos2)
        multiplets = [pos for pos, count in pos_count.items() if count > 1]
        return [(p1, p2) for p1, p2 in zip(pos1, pos2) if not p1 in multiplets and not p2 in multiplets]

    def pairs2bpseq(self, pairs, seq):
        opener = {p[0]+1: p[1]+1 for p in pairs}
        closer = {}
        with open(self.bpseq_path, 'w+') as f:
            for i, nuc in enumerate(seq, 1):
                if i in opener.keys():
                    f.write(str(i) + '\t' + nuc + '\t' + str(opener[i]) + '\n')
                    closer[opener[i]] = i
                    del opener[i]
                elif i in closer.keys():
                    f.write(str(i) + '\t' + nuc + '\t' + str(closer[i]) + '\n')
                    del closer[i]
                else:
                    f.write(str(i) + '\t' + nuc + '\t' + str(0) + '\n')


    def pairs_from_db(self, structure, start_index=0):
        level_stacks = defaultdict(list)
        closing_partners = {')': '(', ']': '[', '}': '{', '>': '<'}
        levels = {')': 0, ']': 1, '}': 2, '>': 3}

        pairs = []

        for i, sym in enumerate(structure, start_index):
            if sym == '.':
                continue
            # high order pks are alphabetical characters
            if sym.isalpha():
                if sym.isupper():
                    level_stacks[sym].append(i)
                else:
                    op = level_stacks[sym.upper()].pop()
                    pairs.append((op, i, ord(sym.upper()) - 61))  # use asci code if letter is used to asign PKs, start with level 4 (A has asci code 65)
            else:
                if sym in closing_partners.values():
                    level_stacks[sym].append(i)
                else:
                    op = level_stacks[closing_partners[sym]].pop()
                    pairs.append((op, i, levels[sym]))
        return pairs


    def parse_st_output(self):
        try:
            with open(self.st_file) as f:
                lines = f.readlines()

        except pd.errors.ParserError:
            print("# Read st ERROR: ParserError st file", self.st_file)
            return False

        lines_plain = []
        for line in lines:
            if not line.startswith('#'):
                lines_plain.append(line)
        lines = lines_plain

        struct = [a for a in lines[1] if a != '\n']
        return ''.join(struct)



    def get_pairs(self, sequence, pos1, pos2):
        all_pairs = [(p1, p2) for p1, p2 in zip(pos1, pos2)]
        multiplets_free_pairs = self.remove_multiplets(pos1, pos2)
        self.pairs2bpseq(multiplets_free_pairs, sequence)
        e_code = self.run()
        if e_code != 0:
            return None
        structure = self.parse_st_output()
        multiplets_free_pairs = self.pairs_from_db(structure)
        # print(all_pairs)
        # print(multiplets_free_pairs)
        pairs = self.get_pair_levels(multiplets_free_pairs, all_pairs)
        return pairs


    def get_pair_levels(self, canonicals, all_pairs, helix_data=None):
        all_pairs = sorted(all_pairs, key=lambda x: (x[0], x[1]))
        if not canonicals:
            if all_pairs:
                canonicals.append((all_pairs[0][0], all_pairs[0][1], 0))
            else:
                return []
        canonical_levels = defaultdict(list)
        canonical_pairs = []
        unassigned_pairs = all_pairs

        helix_ids = []
        if helix_data:
            for _, ids in helix_data.items():
                helix_ids += ids

        # c = [(pair[0], pair[1]) for pair in canonicals]

        for pair in canonicals:
            canonical_levels[pair[2]].append((pair[0], pair[1]))
            # print(pair)
            # print(unassigned_pairs)
            unassigned_pairs.remove((pair[0], pair[1]))


        for pair in unassigned_pairs:
            assigned = False

            for i in range(max(canonical_levels.keys())+1):

                # print(i)
                # canonical_levels[i] = sorted(canonical_levels[i], key=lambda x: x[0], reverse=True)
                opener = [x[0] for x in canonical_levels[i]]
                closer = [x[1] for x in canonical_levels[i]]

                # multiplet handling: If pair in helix_data (coaxial stacking), then it might get assigned to current level
                if pair[0] in opener or pair[0] in closer or pair[1] in opener or pair[1] in closer:
                    if i == 0:
                        if helix_data:
                            assignable = False
                            for _, v in helix_data.items():
                                if (pair[0] in v and pair[1] in v):
                                    assignable = True
                                    break
                        else:
                            assignable = True

                        if not assignable:
                            continue

                if opener and min(opener) >= pair[1] and min(opener) > pair[0]:  # before current nestings
                    # print('before')
                    # print(canonical_levels[i])
                    canonical_levels[i].append(pair)
                    assigned = True
                    break
                elif closer and opener and max(closer) <= pair[1] and min(opener) >= pair[0]:  # surrounds current nestings
                    # print('surrounds')
                    # print(canonical_levels[i])
                    canonical_levels[i].append(pair)
                    assigned = True
                    break
                elif closer and max(closer) <= pair[0] and max(closer) < pair[1]:  # after current nestings
                    # print('after')
                    # print(canonical_levels[i])
                    canonical_levels[i].append(pair)
                    assigned = True
                    break
                else:
                    # none of level pairs is crossed by the current pair (allows triples)
                    enclosed_openers = [p[0] for p in canonical_levels[i] if pair[0] <= p[0] < pair[1]]
                    enclosed_closers = [p[1] for p in canonical_levels[i] if pair[0] < p[1] <= pair[1]]
                    # print(pair)
                    # print(canonical_levels[i])
                    # print(len(enclosed_openers), len(enclosed_closers))
                    if len(enclosed_openers) == len(enclosed_closers):
                        canonical_levels[i].append(pair)
                        assigned = True
                        break
                    else:
                        continue

            if not assigned:
                canonical_levels[max(canonical_levels.keys()) + 1].append(pair)

        all_pairs = []
        for level, pairs in canonical_levels.items():
            all_pairs += [(pair[0], pair[1], level) for pair in pairs]

        return all_pairs
