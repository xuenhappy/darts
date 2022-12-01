#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: enxu
'''
from .utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Embeder(nn.Module):
    """单词表示"""

    def __init__(self, token_size, hidden_size):
        super().__init__()
        self.embeding = nn.Embedding(token_size, hidden_size)
        self.normal = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.5)

    def forward(self, input_idx, type_embeding=None, input_mask=None):
        embeding = self.embeding(input_idx)
        if type_embeding is not None:
            embeding += type_embeding
        embeding = self.drop(self.normal(embeding))
        if input_mask is not None:
            embeding = embeding * input_mask.to(embeding.dtype)
        return embeding


class SentenceEncoder(nn.Module):
    """句子表示"""

    def __init__(self, token_size, hidden_size, type_size=-1):
        super().__init__()
        self.embeding = Embeder(token_size, hidden_size)
        if type_size > 0:
            self.type_embeding = nn.Embedding(type_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size * 2, batch_first=True, bidirectional=True)
        self.dropin = nn.Dropout(0.3)
        self.imner = nn.Linear(hidden_size * 2, hidden_size)
        self.imgate = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.normal = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_idx, seq_lengths=None, type_idxs=None):
        attention_mask = None
        if seq_lengths is not None:
            _tmp_tensor = torch.cumsum(torch.ones(input_idx.size()), dim=-1).to(input_idx.device)
            attention_mask = (_tmp_tensor < seq_lengths.unsqueeze(-1)).byte().unsqueeze(2)

        type_embeding = None
        if (type_idxs is not None) and hasattr(self, 'type_embeding'):
            type_embeding = self.type_embeding(type_idxs)

        embeding = self.embeding(input_idx, type_embeding, attention_mask)
        if seq_lengths is None:
            encoding = self.encoder(embeding)
        else:
            encoding = run_rnn(self.encoder, embeding, seq_lengths)

        encoding = self.dropin(encoding)
        output = self.output(self.imner(encoding) * torch.sigmoid(self.imgate(encoding)))
        output = self.normal(output)
        if attention_mask is not None:
            output = output * attention_mask.to(output.dtype)
        return self.drop(output)


class Quantizer(nn.Module):
    """距离量化模型"""

    def __init__(self, input_size, hidden_size):
        nn.Module.__init__(self)
        self.Kmap = nn.Linear(input_size, hidden_size)
        self.Qmap = nn.Linear(input_size, hidden_size)
        self.alpha = np.sqrt(hidden_size)

    def distance(self, x, y):
        K, Q = self.Kmap(x), self.Qmap(y)
        dist = torch.einsum('ij,ij->i', K, Q) / self.alpha
        return F.silu(dist).squeeze(-1)


class GraphTrainer(nn.Module):
    """图训练"""

    def __init__(self, token_size, hidden_size, type_size=-1):
        super().__init__()
        self.predictor = SentenceEncoder(token_size, hidden_size, type_size=type_size)
        self.quantizer = Quantizer(hidden_size, 32)
        self.lossfunc = GraphLoss()

    def loss(self, batch_input_idx, batch_seq_lengths, batch_type_idxs, batch_atoms, batch_graph, graph_index):
        sentence_embeding = self.predictor(batch_input_idx, batch_seq_lengths, batch_type_idxs)
        atoms_embeding = batch_segment_max(sentence_embeding, batch_atoms)
        batch_atom_index = torch.from_numpy(graph[:, :2]).long().to(atoms_embeding.device)
        losses = []
        gs = 0
        for ge in graph_index:
            graph = batch_graph[gs:ge]
            atom_index = batch_atom_index[gs:ge]
            gs = ge
            s_atom_embeding = atoms_embeding[atom_index[:, 0]]
            e_atom_embeding = atoms_embeding[atom_index[:, 1]]
            weight = self.quantizer(s_atom_embeding, e_atom_embeding)
            losses.append(self.lossfunc(graph, weight))
        return sum(losses) / len(losses)

    def forward(self, batch_input_idx, batch_seq_lengths, batch_type_idxs, batch_atoms, batch_graph, graph_index):
        """

        Args:
            batch_input_idx (int numpy): N*T
            batch_seq_lengths (int numpy): N
            batch_type_idxs (int numpy): N*T
            batch_atoms (int numpy): M*3 coloum is [batch_idx start,end]
            batch_graph (int numpy): K*3 coloum is [atom_sidx,atom_eidx,bool]
            graph_index (int numpy): N coloum is [graph_eidx]

        Returns:
            losses
        """

        batch_input_idx = torch.from_numpy(batch_input_idx).long()
        batch_seq_lengths = torch.from_numpy(batch_seq_lengths).long()
        batch_type_idxs = torch.from_numpy(batch_type_idxs).long()
        batch_atoms = torch.from_numpy(batch_atoms).long()

        if torch.cuda.is_available():
            batch_input_idx = batch_input_idx.cuda()
            batch_seq_lengths = batch_seq_lengths.cuda()
            batch_type_idxs = batch_type_idxs.cuda()
            batch_atoms = batch_atoms.cuda()

        return self.loss(self, batch_input_idx, batch_seq_lengths, batch_type_idxs, batch_atoms, batch_graph,
            graph_index)
