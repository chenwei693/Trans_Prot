import pandas as pd
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

import torch
from torch.utils.data import TensorDataset, DataLoader



import numpy as np



def position_encoding(seqs):

    d = 128
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return np.array(res)


def data_construct(seqs, file_prot, labels, train):
    df = pd.read_csv(file_prot)
    has_label = "label" in df.columns
    features = df.drop(columns=["label"]).values if has_label else df.values  # shape: [num, seq_len * dim]
    # print(features.shape)
    num_samples, total_dim = features.shape
    dim = 1024
    seq_len = total_dim // dim
    prot_tensor = torch.tensor(features, dtype=torch.float32).reshape(num_samples, seq_len, dim)

    longest_num = len(max(seqs, key=len))
    # print(f'longest_num:{longest_num}')
    seqs = [i.ljust(longest_num, 'X') for i in seqs]

    aa_dict = { 'X': 0,'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22}
    pos_embed = position_encoding(seqs)

    pep_codes = []
    for pep in seqs:
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa.upper()])
        pep_codes.append((torch.tensor(current_pep)).cuda())

    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True).cuda()

    dataset = Data.TensorDataset(embed_data, torch.FloatTensor(pos_embed).cuda(),
                                 prot_tensor.cuda(),
                                 torch.LongTensor(labels).cuda())
    batch_size = 128
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_iter



def load_bench_data(file, file_prot):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, file_prot, labels, train=True)
    data_iter = list(data_iter)
    train_iter = [x for i, x in enumerate(data_iter) if i % 5 != 0]
    test_iter = [x for i, x in enumerate(data_iter) if i % 5 == 0]

    return train_iter, test_iter


def load_ind_data(file, prot_file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, prot_file, labels,
                               train=False)
    return data_iter


def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cuda'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model
