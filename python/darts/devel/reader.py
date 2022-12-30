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

import numpy as np


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


class LineSampleReader():

    def __init__(self, filep):
        self.samplefile = filep

    def process(self, line):
        raise Exception("not impl")

    def sample_iter(self):
        with open(self.samplefile, encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                yield line

    def __iter__(self):
        if self.processNum < 1:
            for line in self.sample_iter():
                result = self.process(line)
                if not result:
                    continue
                yield result
        else:
            for result in MutiProcessIter(self.sample_iter(), self.process, self.processNum):
                if not result:
                    continue
                yield result
