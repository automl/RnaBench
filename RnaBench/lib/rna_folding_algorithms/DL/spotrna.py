import warnings
warnings.filterwarnings('ignore')

import pickle
import os, six, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '2'
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import os, six, sys, time

from pathlib import Path

class SpotRna():
    def __init__(
                 self,
                 base_dir='.',
                 data_dir='data',
                 gpu=-1,
                 cpu=1,
    ):

        self.tfr_path = os.path.join(base_dir, 'external_algorithms', 'SPOT-RNA', 'spotrna.tfrecords')
        self.spot_path = os.path.join(base_dir, 'external_algorithms', 'SPOT-RNA')
        self.gpu = gpu
        self.cpu = cpu

    def __name__(self):
        return 'SPOT-RNA'

    def __repr__(self):
        return 'SPOT-RNA'

    def __call__(self, sequence):
        preparing_tfr_records(sequence, self.tfr_path)
        pairs = infer_spotrna(sequence,
                              self.tfr_path,
                              self.spot_path,
                              gpu=self.gpu,
                              cpu=self.cpu,
                              )
        Path(self.tfr_path).unlink()
        return pairs


def run_spotrna(sequence,
                 base_dir = '.',
                 data_dir = 'data',
                 gpu=-1,
                 cpu=1):

    tfr_path = os.path.join(base_dir, 'external_algorithms', 'SPOT-RNA', 'spotrna.tfrecords')
    spot_path = os.path.join(base_dir, 'external_algorithms', 'SPOT-RNA')

    preparing_tfr_records(sequence, tfr_path)
    pairs = infer_spotrna(sequence, tfr_path, spot_path, gpu=gpu, cpu=cpu)

    return pairs


def infer_spotrna(sequence, tfr_path, base_dir, gpu=-1, cpu=1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    NUM_MODELS = 5
    test_loc = [os.path.join(tfr_path)]
    outputs = {}
    mask = {}
    count = 1  # df.shape[0]
    def sigmoid(x):
        return 1/(1+np.exp(-np.array(x, dtype=np.float128)))
    for MODEL in range(NUM_MODELS):

        if gpu==-1:
                config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=cpu, inter_op_parallelism_threads=cpu)
        else:
    	    config = tf.compat.v1.ConfigProto()
    	    config.allow_soft_placement=True
    	    config.log_device_placement=False
        with tf.compat.v1.Session(config=config) as sess:
            saver = tf.compat.v1.train.import_meta_graph(os.path.join(base_dir, 'SPOT-RNA-models', 'model' + str(MODEL) + '.meta'))
            saver.restore(sess,os.path.join(base_dir, 'SPOT-RNA-models', 'model' + str(MODEL)))
            graph = tf.compat.v1.get_default_graph()
            init_test =  graph.get_operation_by_name('make_initializer_2')
            tmp_out = graph.get_tensor_by_name('output_FC/fully_connected/BiasAdd:0')
            name_tensor = graph.get_tensor_by_name('tensors_2/component_0:0')
            RNA_name = graph.get_tensor_by_name('IteratorGetNext:0')
            label_mask = graph.get_tensor_by_name('IteratorGetNext:4')
            sess.run(init_test,feed_dict={name_tensor:test_loc})
            # pbar = tqdm(total = count)
            while True:
                try:
                    out = sess.run([tmp_out,RNA_name,label_mask],feed_dict={'dropout:0':1})
                    out[1] = out[1].decode()
                    mask[out[1]] = out[2]

                    if MODEL == 0:
                        outputs[out[1]] = [sigmoid(out[0])]
                    else:
                        outputs[out[1]].append(sigmoid(out[0]))
                    # pbar.update(1)
                except:
                    break
            # pbar.close()
        tf.compat.v1.reset_default_graph()

    RNA_ids = [i for i in list(outputs.keys())]
    i = RNA_ids[0]
    ensemble_outputs = {}
    seq = ''.join(sequence)
    ensemble_outputs[i] = np.mean(outputs[i],0)
    pred_pairs, save_multiplets, non_canonicals, lone_pairs, seq, name = get_pairs(ensemble_outputs[i], mask[i], seq, i)
    pairs = pred_pairs + save_multiplets + non_canonicals + lone_pairs
    pairs = [[p1, p2, 0] for p1, p2 in pairs]  # we don't care about the page number here...
    return pairs


def preparing_tfr_records(sequence, tfr_path, id=0):
    # print('\nPreparing tfr records file for SPOT-RNA:')
    tfr_path = tfr_path
    seq_len, feature, zero_mask, label_mask, true_label = get_data(''.join(sequence))

    with tf.io.TFRecordWriter(tfr_path) as writer:
        feat = tf.train.Example(features=tf.train.Features(feature={'rna_name': _bytes_feature(str(id)),
                                                                    'seq_len': _int64_feature(seq_len),
                                                                    'feature': _float_feature(feature),
                                                                    'zero_mask': _float_feature(zero_mask),
                                                                    'label_mask': _float_feature(label_mask),
                                                                    'true_label': _float_feature(true_label),
                                                                    }))

        writer.write(feat.SerializeToString())
    writer.close()


def get_pair_levels(canonicals, all_pairs, helix_data=None):
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


def save_df(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)


def load_df(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            df = pickle.load(f)
        return df
    else:
        return None


# import from SPOT-RNA.utils.utils
def get_data(seq):
    seq_len = len(seq)
    one_hot_feat = one_hot(seq)
    #print(one_hot_feat[-1])
    zero_mask = z_mask(seq_len)[None, :, :, None]
    label_mask = l_mask(one_hot_feat, seq_len)
    temp = one_hot_feat[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    #out = true_output

    return seq_len, [i for i in (feature.astype(float)).flatten()], [i for i in zero_mask.flatten()], [i for i in label_mask.flatten()], [i for i in label_mask.flatten()]

# import from SPOT-RNA.utils.utils
def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)

# import from SPOT-RNA.utils.utils
def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])

    return feat

# import from SPOT-RNA.utils.utils
def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)

# import from SPOT-RNA.utils.utils
def _int64_feature(value):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# import from SPOT-RNA.utils.utils
def _float_feature(value):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# import from SPOT-RNA.utils.utils
def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# import from SPOT-RNA.utils.utils
def output_mask(seq, NC=True):
    if NC:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'CC', 'GG', 'AG', 'CA', 'AC', 'UU', 'AA', 'CU', 'GA', 'UC']
    else:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
    mask = np.zeros((len(seq), len(seq)))
    for i, I in enumerate(seq):
        for j, J in enumerate(seq):
            if str(I) + str(J) in include_pairs:
                mask[i, j] = 1
    return mask

# import from SPOT-RNA.utils.utils
def multiplets_free_bp(pred_pairs, y_pred):
    L = len(pred_pairs)
    multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = []
    while len(multiplets_bp) > 0:
        remove_pairs = []
        for i in multiplets_bp:
            save_prob = []
            for j in i:
                save_prob.append(y_pred[j[0], j[1]])
            remove_pairs.append(i[save_prob.index(min(save_prob))])
            save_multiplets.append(i[save_prob.index(min(save_prob))])
        pred_pairs = [k for k in pred_pairs if k not in remove_pairs]
        multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = [list(x) for x in set(tuple(x) for x in save_multiplets)]
    assert L == len(pred_pairs)+len(save_multiplets)
    #print(L, len(pred_pairs), save_multiplets)
    return pred_pairs, save_multiplets

# import from SPOT-RNA.utils.utils
def multiplets_pairs(pred_pairs):

    pred_pair = [i[:2] for i in pred_pairs]
    temp_list = flatten(pred_pair)
    temp_list.sort()
    new_list = sorted(set(temp_list))
    dup_list = []
    for i in range(len(new_list)):
        if (temp_list.count(new_list[i]) > 1):
            dup_list.append(new_list[i])

    dub_pairs = []
    for e in pred_pair:
        if e[0] in dup_list:
            dub_pairs.append(e)
        elif e[1] in dup_list:
            dub_pairs.append(e)

    temp3 = []
    for i in dup_list:
        temp4 = []
        for k in dub_pairs:
            if i in k:
                temp4.append(k)
        temp3.append(temp4)

    return temp3

# import from SPOT-RNA.utils.utils
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]],sequence[i[1]]] in [["A","U"], ["U","A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]],sequence[i[1]]] in [["G","C"], ["C","G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]],sequence[i[1]]] in [["G","U"], ["U","G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
        # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def lone_pair(pairs):
    lone_pairs = []
    pairs.sort()
    for i, I in enumerate(pairs):
        if ([I[0] - 1, I[1] + 1] not in pairs) and ([I[0] + 1, I[1] - 1] not in pairs):
            lone_pairs.append(I)

    return lone_pairs


def id2mat(pos1id, pos2id, max_id):
    mat = np.zeros([max_id, max_id])
    for i1, i2 in zip(pos1id, pos2id):
        mat[i1, i2] = 1
        mat[i2, i1] = 1
    return mat



def df2fasta(df, path):
    with open(path, 'w+') as f:
        for i, row in df.iterrows():
            f.write('>'+str(i)+'\n'+''.join(row['sequence'])+'\n')


def parse_intermediate_output(data, intermediate_output, algorithm):
    with open(intermediate_output, 'rb') as f:
        lines = [line.decode('utf-8').strip() for line in f.readlines()]

    # print(lines)

    ids = []
    preds = []
    for i, line in enumerate(lines):
        if i % 3 == 0:
            ids.append(int(line[1:]))
        elif i % 3 == 2:
            preds.append(line.split()[0])

    predictions = {}
    for i, pred in zip(ids, preds):
        pairs = pairs_from_db(pred)
        pos1id = [p[0]-1 for p in pairs]
        pos2id = [p[1]-1 for p in pairs]
        pk = [p[2] for p in pairs]
        sample = {f"{algorithm}_pos1id": pos1id,
                  f"{algorithm}_pos2id": pos2id,
                  f"{algorithm}_pk": pk,
                  'str_seq': data.loc[i, 'str_seq']}

        predictions[i] = sample

    predictions_df = pd.DataFrame(predictions).T
    predictions = data.reset_index().merge(predictions_df, how='left', on=['str_seq']).set_index('index')

    predictions = predictions.drop(['str_seq'], axis=1)

    return predictions




def get_pairs(ensemble_outputs, label_mask, seq, name):
    Threshold = 0.335
    test_output = ensemble_outputs
    mask = output_mask(seq)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    for i in range(test_output.shape[0]):
        y_pred[inds[0][i], inds[1][i]] = test_output[i]
    y_pred = np.multiply(y_pred, mask)

    tri_inds = np.triu_indices(y_pred.shape[0], k=1)

    out_pred = y_pred[tri_inds]
    outputs = out_pred[:, None]
    seq_pairs = [[tri_inds[0][j], tri_inds[1][j], ''.join([seq[tri_inds[0][j]], seq[tri_inds[1][j]]])] for j in
                 range(tri_inds[0].shape[0])]

    outputs_T = np.greater_equal(outputs, Threshold)
    pred_pairs = [i for I, i in enumerate(seq_pairs) if outputs_T[I]]
    pred_pairs = [i[:2] for i in pred_pairs]
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, y_pred)

    watson_pairs, wobble_pairs, noncanonical_pairs = type_pairs(pred_pairs, seq)
    lone_bp = lone_pair(pred_pairs)

    return pred_pairs, save_multiplets, noncanonical_pairs, lone_bp, seq, name
