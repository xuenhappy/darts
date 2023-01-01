'''
File: reader.py
Project: devel
File Created: Saturday, 8th January 2022 9:12:42 pm
Author: Xu En (xuen@mokar.com)
-----
Last Modified: Saturday, 8th January 2022 9:13:03 pm
Modified By: Xu En (xuen@mokahr.com)
-----
Copyright 2021 - 2022 Your Company, Moka
'''
import torch
import numpy as np
import torch.nn as nn
from ..cdarts import *


def d2list2array(d2list, fval=0, dtype=np.int32):
    """
    convert 2d length no eq array  arralist to a 2d matrix
    """
    lens = np.asarray(list(len(item) for item in d2list), np.int32)
    mask = lens[:, None] > np.arange(lens.max())
    out = np.ones(mask.shape, dtype=dtype) * fval
    out[mask] = np.concatenate(d2list)
    return out, lens


def wordlist2graph(wordarr):
    #word add is a int tensor like batch*(bixd,atom_s,atome_e,bool)
    #return is batch*(gbidx,w_s,w_e,bool)
    feats_lens = np.unique(wordarr[:, 0], return_counts=True)[1]
    s, e = 0, 0
    graphs = []
    for gidx, lens in enumerate(feats_lens):
        e = s + lens
        words = wordarr[s:e]
        advj_ = (words[:, 2].reshape(-1, 1) == words[:, 1].reshape(1, -1))
        best_ = (words[:, 3].reshape(-1, 1) * words[:, 3].reshape(1, -1))
        node_index = np.arange(lens) + s
        rowidx = node_index.reshape(-1, 1).repeat(lens, 1)
        colidx = node_index.reshape(1, -1).repeat(lens, 0)
        idxes = np.stack((rowidx, colidx, best_), 2)[advj_]
        bidxs = np.ones(idxes.shape[0]).reshape(-1, 1) * gidx
        graphs.append(np.concatenate((bidxs, idxes), 1))
        s = e
    return np.concatenate(graphs, 0)


class TokenNerSampleReader():

    def __init__(self, filep, max_sent_asize=50):
        self.samplefile = filep
        self.aencoder = AtomCodec({"base.dir": "data/models/codex"})
        self.max_sent_asize = max_sent_asize

    def decode(self, codes):
        return " ".join(self.aencoder.decode(l) for l in codes)

    def _sample_iter(self):
        with open(self.samplefile, encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                if len(line) < 3:
                    continue
                yield line

    def getBio(self, line):
        line = normalize(line).strip()
        tokens = line.split(" ")
        line_str = [tokens[0]]
        for word in tokens[1:]:
            if not word:
                continue
            if ord(word[0]) < 255 and ord(line_str[-1][-1]) < 255:
                line_str.append(" ")
            line_str.append(word)
        bios = []
        for word in line_str:
            if len(word) > 1 and any(ord(x) > 255 for x in word):
                bio = ['I-_HWORD'] * len(word)
                bio[0] = 'B-_HWORD'
            else:
                bio = ['O'] * len(word)
            bios.extend(bio)

        alist = PyAtomList("".join(line_str), normal_before=False)
        abio = ['O']  #append 'START'
        for atom in alist.tolist():
            tag = set(bios[atom.st:atom.et])
            if len(tag) == 1:
                if 'I-_HWORD' in tag:
                    abio.append('I-_HWORD')
                elif 'B-_HWORD' in tag:
                    abio.append('B-_HWORD')
                else:
                    abio.append('O')
            else:
                if 'B-_HWORD' in tag:
                    abio.append('B-_HWORD')
                elif 'I-_HWORD' in tag:
                    abio.append('I-_HWORD')
                else:
                    abio.append('O')
        abio.append('O')  #appaned end
        return alist, abio

    def __iter__(self):
        for line in self._sample_iter():
            alist, bios = self.getBio(line)
            codes = self.aencoder.encode(alist)
            idxs = [code[0] for code in codes]
            atom_code_s = [0] * len(bios)
            atom_code_e = [0] * len(bios)
            atom_code_s[-1] = len(codes) - 1
            atom_code_e[-1] = len(codes) - 1
            for i, code in enumerate(codes):
                index = code[1]
                if index < 0:
                    continue
                index += 1
                if atom_code_s[index] == 0:
                    atom_code_s[index] = i
                atom_code_e[index] = i
            print("----------------------------------------------")
            print(bios)
            print(atom_code_s)
            print(atom_code_e)
            print(self.decode(idxs))
