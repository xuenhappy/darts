#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: enxu
'''
import utils
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class SMelo(nn.Module):
    def __init__(self, token_size):
        super().__init__()
        self.embeding = nn.Embedding(token_size, 150)
        self.drop = nn.Dropout(0.5)
        self.rnn = nn.GRU(150, 180, batch_first=True, bidirectional=True)
        self.map = nn.Linear(180*2, 150)
        self.bd = nn.BatchNorm1d(150)

    def forward(self, input, seq_lengths, input_add=None):
        """
        input is a long tensor and shape is (N,T),
        seq_lengths is a long tensor and shape is (N,)
        seq_length and inpt must be in same device
        """
        embeding = self.drop(self.embeding(input))
        if input_add is not None:
            embeding += input_add
        out = self.map(run_rnn(self.rnn, embeding, seq_lengths))+embeding
        return self.bd(out.view(-1, out.size(2))).view(out.shape)


class Quantizer(nn.Module):

    def __init__(self, input_size, hidden_size):
        nn.Module.__init__(self)
        self.Kmap = nn.Linear(input_size, hidden_size)
        self.Qmap = nn.Linear(input_size, hidden_size)
        self.alpha = np.sqrt(hidden_size)

    def distance(self, x, y):
        K, Q = self.Kmap(x), self.Qmap(y)
        dist = torch.einsum('ij,ij->i', K, Q)/self.alpha
        return F.softplus(dist)


class SMeloTrainer(nn.Module):
    """
    init the sentence
    """

    def __init__(self, word_num, emb_size, tags_weight, nsampled):
        super(SentencePredict, self).__init__()
        self.predictor = SentenceEncoder(word_num, emb_size)
        self.embedd = utils.SampledSoftMaxCrossEntropy(emb_size, tags_weight, nsampled)

    def forward(self, batch_sentence, batch_tags_with_idx, keep_prop):
        """
        batch_sentence shape:batch_size×word_size
        batch_tags_with_idx:list((start,end,tag))
        """
        batch_sentence_length = []
        batch_indices_st = []
        batch_indices_et = []
        batch_tags = []

        for i, batch in enumerate(batch_tags_with_idx):
            offset = i*batch_sentence.shape[1]
            for s, e, tag in batch:
                batch_indices_st.append(offset+s)
                batch_indices_et.append(offset+e)
                batch_tags.append(batch_tags)
            batch_sentence_length.append(batch[-1][1])

        batch_sentence = torch.from_numpy(batch_sentence)
        batch_sentence_length = torch.LongTensor(batch_sentence_length)
        batch_indices_st = torch.LongTensor(batch_indices_st)
        batch_indices_et = torch.LongTensor(batch_indices_et)
        batch_tags = torch.LongTensor(batch_tags)

        if torch.cuda.is_available():
            batch_sentence = batch_sentence
            batch_sentence_length = batch_sentence_length.cuda()
            batch_indices_st = batch_indices_st.cuda()
            batch_indices_et = batch_indices_et.cuda()
            batch_tags = batch_tags.cuda()

        batch_sentence_fw_encoding, batch_sentence_bw_encoding = self.predictor(batch_sentence, batch_sentence_length, keep_prop)
        word_embs_st = batch_sentence_fw_encoding.reshape(-1, emb_size).index_select(0, batch_indices_st)
        word_embs_et = batch_sentence_bw_encoding.reshape(-1, emb_size).index_select(0, batch_indices_et)
        word_embs = word_embs_st + word_embs_et
        return self.embedd(word_embs, batch_tags).mean()
