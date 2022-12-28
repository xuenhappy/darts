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
from .utils import *


class WordEncoder(nn.Module):

    def __init__(self, vocab_num, vocab_esize, hidden_size, word_esize, wtype_num):
        super().__init__()
        self.vocab_num = vocab_num
        self.wtype_num = wtype_num
        self.vocab_embeding = nn.Embedding(vocab_num, vocab_esize)
        self.encoder = nn.GRU(vocab_esize, hidden_size, batch_first=True, bidirectional=True)
        if wtype_num > 0:
            self.dropin = nn.Dropout(0.2)
            self.type_embeding = nn.Embedding(wtype_num, word_esize)
        self.imner = nn.Linear(hidden_size * 2, word_esize)

        self.normal = nn.LayerNorm(hidden_size)

    def forward(self, batch_input_idx, batch_word_info):
        #batch_input_idx (batch*time_step)
        #batch_word_info(words_num*[bidx,s,e,tidx])
        sent_embeding = self.imner(self.encoder(self.vocab_embeding(batch_input_idx)))
        word_head_embeding = sent_embeding[batch_word_info[:, 0], batch_word_info[:, 1]]
        word_tail_embeding = sent_embeding[batch_word_info[:, 0], batch_word_info[:, 2]]
        word_sent_embeding = (word_head_embeding + word_tail_embeding) / 2.0
        if self.wtype_num > 0:
            word_type_embeding = self.type_embeding(batch_word_info[:, 3])
            return self.normal(self.dropin(word_type_embeding) + word_sent_embeding)
        return self.normal(word_sent_embeding)

    def export2onnx(self):
        sents_idx = torch.randint(0, self.vocab_num, (11, ))
        word_se = torch.LongTensor([[0, 0], [1, 2], [3, 3], [4, 6], [7, 7], [8, 9], [10, 10]])
        wtype_idx = torch.randint(0, self.wtype_num, (word_se.shape[0], 1))
        word_info = torch.concat((word_se, wtype_idx), dim=1)

        def _script(sents, words):
            sents = torch.unsqueeze(sents, 0)
            bidx = torch.zeros((words.shape[0], 1), dtype=wtype_idx.dtype)
            words = torch.concat((bidx, words), dim=1)
            return self(sents, words)

        # args must same as forawrd
        outfile = "lstm.encoder.onnx"
        inputdata = (sents_idx, word_info)
        inputnames = ['sents', 'wordinfo']
        dynamic_axes = {"sents": {0: 'timestep'}, "wordinfo": {0: 'wordnums'}, "wordemb": {0: 'wordnums'}}

        torch.onnx.export(
            _script,
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['wordemb'],  # the model's output names
            dynamic_axes=dynamic_axes)
        return outfile


class Quantizer(nn.Module):

    def __init__(self, input_size, hidden_size):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kmap = nn.Linear(input_size, hidden_size)
        self.Qmap = nn.Linear(input_size, hidden_size)
        self.alpha = np.sqrt(hidden_size)

    def forward(self, x, y):
        K, Q = self.Kmap(x), self.Qmap(y)
        dist = torch.einsum('ij,ij->i', K, Q) / self.alpha
        return F.softplus(dist).view(-1)

    def export2onnx(self):

        def _script(x, y):
            return self(torch.unsqueeze(x, 0), torch.unsqueeze(y, 0))

        # args must same as forawrd
        outfile = "sample.quantizer.onnx"
        inputdata = (torch.randn((self.input_size, )), torch.randn((self.input_size, )))
        inputnames = ['a', 'b']
        dynamic_axes = {}

        torch.onnx.export(
            _script,
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['distance'],  # the model's output names
            dynamic_axes=dynamic_axes)
        return outfile


class CrfNer(nn.Module):

    def __init__(self, vocab_num, vocab_esize, hidden_size, word_esize, tag_nums):
        super().__init__()
        self.encoder = WordEncoder(vocab_num, vocab_esize, hidden_size, word_esize, -1)
        self.prop = nn.Linear(word_esize, tag_nums)
        self.crf = CRFLoss(tag_nums)

    def getWordprop(self, batch_input_idx, batch_word_info):
        #batch_input_idx (batch*time_step)
        #batch_word_info(words_num*[bidx,s,e])
        return self.prop(self.encoder(batch_input_idx, batch_word_info))

    def forward(self, batch_input_idx, batch_word_info):
        #batch_input_idx (batch*time_step)
        #batch_word_info(words_num*[bidx,s,e,tagidx])
        wordprop = self.getWordprop(batch_input_idx, batch_word_info)

        featsLen, featsIdx = getFeatsIdx(batch_word_info[:, 0])
        word_sent = wordprop.index_select(0, featsIdx.view(-1)).view((*featsIdx.shape, -1))
        tag_sent = batch_word_info[:, -1].index_select(0, featsIdx.view(-1)).view(featsIdx.shape)
        return self.crf(word_sent, tag_sent, featsLen)

    def export2onnx(self):
        sents_idx = torch.randint(0, self.vocab_num, (11, ))
        word_info = torch.LongTensor([[0, 0], [1, 2], [3, 3], [4, 6], [7, 7], [8, 9], [10, 10]])

        def _script(sents, words):
            sents = torch.unsqueeze(sents, 0)
            bidx = torch.zeros((words.shape[0], 1), dtype=words.dtype)
            words = torch.concat((bidx, words), dim=1)
            prop = self.getWordprop(sents, words)
            return self.crf.viterbi_decode(prop)

        # args must same as forawrd
        outfile = "lstm.encoder.onnx"
        inputdata = (sents_idx, word_info)
        inputnames = ['sents', 'wordinfo']
        dynamic_axes = {"sents": {0: 'timestep'}, "wordinfo": {0: 'wordnums'}, "wordemb": {0: 'wordnums'}}

        torch.onnx.export(
            _script,
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['wordemb'],  # the model's output names
            dynamic_axes=dynamic_axes)
        return outfile


class GraphTrainer(nn.Module):

    def __init__(self, vocab_num, vocab_esize, hidden_size, word_esize, wtype_num):
        super().__init__()
        self.predictor = WordEncoder(vocab_num, vocab_esize, hidden_size, word_esize, wtype_num)
        self.quantizer = Quantizer(hidden_size, 32)
        self.lossfunc = GraphLoss()

    def loss(self, batch_input_idx, batch_word_info, batch_graph):
        #batch_input_idx (batch*time_step)
        #batch_word_info(words_num*[bidx,s,e,tidx])
        #batch_graph:batch*(bidx,bwidx_s,bwidx_e,bool)
        word_embeds = self.predictor(batch_input_idx, batch_word_info)
        egeds_dists = self.quantizer(word_embeds[batch_graph[:, 1]], word_embeds[batch_graph[:, 2]])
        featsLen, featsIdx = getFeatsIdx(batch_graph[:, 0])
        losses = torch.FloatTensor(0, device=batch_input_idx.device)
        for bidx in range(featsLen.shape[0]):
            nidxs = featsIdx[bidx][:featsLen[bidx]]
            edge_weight = egeds_dists[nidxs]
            graph = batch_graph[nidxs][:, 1:]
            losses += self.lossfunc(graph, edge_weight)

        return losses / featsLen.shape[0]

    def forward(self, batch_input_idx, batch_word_info, batch_graph):
        if torch.cuda.is_available():
            batch_input_idx = batch_input_idx.cuda()
            batch_word_info = batch_word_info.cuda()
            batch_graph = batch_graph.cuda()
        return self.loss(self, batch_input_idx, batch_word_info, batch_graph)
