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


class WordEncoder(nn.Module):

    def __init__(self, vocab_num, hidden_size, wtype_num, num_layers=2, num_heads=4, max_positions=4096):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.vocab_num = vocab_num
        self.wtype_num = wtype_num
        self.max_positions = max_positions
        self.vocab_embeding = nn.Sequential(
            nn.Embedding(vocab_num, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-7),
            nn.Dropout(0.1),
        )
        self.position_embeding = nn.Embedding(max_positions, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 norm=nn.LayerNorm(hidden_size))

        if wtype_num > 0:
            self.type_embeding = nn.Sequential(
                nn.Embedding(wtype_num, hidden_size),
                nn.LayerNorm(hidden_size, eps=1e-7),
                nn.Dropout(0.1),
            )
            self.type_normal = nn.LayerNorm(hidden_size)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info):
        #batch_input_idx (batch*time_step)
        #batch_lengths (batch,)
        #batch_word_info(words_num*[bidx,s,e,tidx])
        steps = batch_input_idx.shape[1]
        positions = torch.arange(steps, device=batch_input_idx.device).clamp_max(self.max_positions - 1)
        vocab_emb = self.vocab_embeding(batch_input_idx) + self.position_embeding(positions).unsqueeze(0)
        padding_mask = positions.unsqueeze(0) >= batch_lengths.unsqueeze(1)
        sent_embeding = self.transformer(vocab_emb, src_key_padding_mask=padding_mask)

        word_head_embeding = sent_embeding[batch_word_info[:, 0], batch_word_info[:, 1]]
        word_tail_embeding = sent_embeding[batch_word_info[:, 0], batch_word_info[:, 2]]
        word_sent_embeding = (word_head_embeding + word_tail_embeding) / 2.0
        if self.wtype_num > 0:
            word_type_embeding = self.type_embeding(batch_word_info[:, 3])
            return self.type_normal(word_sent_embeding + word_type_embeding)
        return word_sent_embeding

    def export2onnx(self):
        sents_idx = torch.randint(0, self.vocab_num, (11, ))
        word_se = torch.LongTensor([[0, 0], [1, 2], [3, 3], [4, 6], [7, 7], [8, 9], [10, 10]])
        wtype_idx = torch.randint(0, self.wtype_num, (word_se.shape[0], 1))
        word_info = torch.concat((word_se, wtype_idx), dim=1)

        class _script(nn.Module):

            def __init__(self, obj) -> None:
                super().__init__()
                self.obj = obj

            def forward(self, sents, words):
                lens = torch.LongTensor([sents.shape[0]]).to(sents)
                bsents = torch.unsqueeze(sents, 0)
                bidx = torch.zeros((words.shape[0], 1), dtype=wtype_idx.dtype)
                words = torch.concat((bidx, words), dim=1)
                return self.obj(bsents, lens, words)

        # args must same as forawrd
        outfile = "transformer.encoder.onnx"
        inputdata = (sents_idx, word_info)
        inputnames = ['sents', 'wordinfo']
        dynamic_axes = {"sents": {0: 'timestep'}, "wordinfo": {0: 'wordnums'}, "wordemb": {0: 'wordnums'}}

        torch.onnx.export(
            _script(self),
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=17,  # Transformer attention export requires a modern opset.
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
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / np.sqrt(hidden_size)), dtype=torch.float32))

    def forward(self, x, y):
        keys = F.normalize(self.Kmap(x), dim=-1)
        queries = F.normalize(self.Qmap(y), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        similarity = torch.sum(keys * queries, dim=-1) * scale
        return F.softplus(-similarity).view(-1)

    def export2onnx(self):
        outfile = "sample.quantizer.onnx"
        inputdata = (torch.randn((1, self.input_size)), torch.randn((1, self.input_size)))
        inputnames = ['a', 'b']
        dynamic_axes = {'a': {0: 'edges'}, 'b': {0: 'edges'}, 'distance': {0: 'edges'}}

        torch.onnx.export(
            self,
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=17,
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['distance'],  # the model's output names
            dynamic_axes=dynamic_axes)
        return outfile


class CrfNer(nn.Module):

    def __init__(self, vocab_num, hidden_size, tag_nums):
        super().__init__()
        self.encoder = WordEncoder(vocab_num, hidden_size, -1)
        self.prop = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, tag_nums),
        )
        self.crf = CRFLoss(tag_nums)

    def getWordprop(self, batch_input_idx, batch_lengths, batch_word_info):
        #batch_input_idx (batch*time_step)
        #batch_lengths (batch,)
        #batch_word_info(words_num*[bidx,s,e])
        sents_emb = self.encoder(batch_input_idx, batch_lengths, batch_word_info)
        return self.prop(sents_emb)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info):
        #batch_input_idx (batch*time_step)
        #batch_lengths (batch,)
        #batch_word_info(words_num*[bidx,s,e,tagidx])
        wordprop = self.getWordprop(batch_input_idx, batch_lengths, batch_word_info)
        featsLen, featsIdx = getFeatsIdx(batch_word_info[:, 0])
        word_sent = wordprop.index_select(0, featsIdx.view(-1)).view((*featsIdx.shape, -1))
        tag_sent = batch_word_info[:, -1].index_select(0, featsIdx.view(-1)).view(featsIdx.shape)
        return self.crf(word_sent, tag_sent, featsLen)

    def export2onnx(self):
        sents_idx = torch.randint(0, self.encoder.vocab_num, (11, ))
        word_info = torch.LongTensor([[0, 0], [1, 2], [3, 3], [4, 6], [7, 7], [8, 9], [10, 10]])

        class _script(nn.Module):

            def __init__(self, obj) -> None:
                super().__init__()
                self.obj = obj

            def forward(self, sents, words):
                lens = torch.LongTensor([sents.shape[0]]).to(sents)
                bsents = torch.unsqueeze(sents, 0)
                bidx = torch.zeros((words.shape[0], 1), dtype=words.dtype)
                bwords = torch.concat((bidx, words), dim=1)
                prop = self.obj.getWordprop(bsents, lens, bwords)
                return self.obj.crf.viterbi_decode(prop)

        # args must same as forawrd
        outfile = "crf.ner.onnx"
        inputdata = (sents_idx, word_info)
        inputnames = ['sents', 'wordinfo']
        dynamic_axes = {"sents": {0: 'timestep'}, "wordinfo": {0: 'wordnums'}, "wordemb": {0: 'wordnums'}}

        torch.onnx.export(
            _script(self),
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=17,
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['wordemb'],  # the model's output names
            dynamic_axes=dynamic_axes)
        return outfile


class NerTrainer(nn.Module):

    def __init__(self, vocab_num, hidden_size, tag_nums):
        super().__init__()
        self.ner = CrfNer(vocab_num, hidden_size, tag_nums)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info):
        if torch.cuda.is_available():
            batch_input_idx = batch_input_idx.cuda()
            batch_word_info = batch_word_info.cuda()
            batch_lengths = batch_lengths.cuda()
        return self.ner(batch_input_idx, batch_lengths, batch_word_info).mean()


class GraphTrainer(nn.Module):

    def __init__(self, vocab_num, hidden_size, wtype_num):
        super().__init__()
        self.predictor = WordEncoder(vocab_num, hidden_size, wtype_num)
        self.quantizer = Quantizer(hidden_size, 32)
        self.lossfunc = GraphLoss()

    def loss(self, batch_input_idx, batch_word_info, batch_graph):
        #batch_input_idx (batch*time_step)
        #batch_word_info(words_num*[word_bidx,input_s,input_e,wtype_idx])
        #batch_graph:batch*(graph_bidx,bwidx_s,bwidx_e,bool)
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
        return self.loss(batch_input_idx, batch_word_info, batch_graph)


class GraphTrainerV2(nn.Module):
    """
    Sparse batched graph trainer.

    batch_graph is expected to be a flat edge table:
    [batch_id, src_node, dst_node, gold_mask]
    where src/dst are flattened node ids aligned with batch_word_info.
    """

    def __init__(self, vocab_num, hidden_size, wtype_num, edge_chunk_size=65536, strict_path=True):
        super().__init__()
        self.predictor = WordEncoder(vocab_num, hidden_size, wtype_num)
        self.quantizer = Quantizer(hidden_size, 32)
        self.lossfunc = GraphLossSparse(edge_chunk_size=edge_chunk_size, strict_path=strict_path)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info, batch_graph):
        if torch.cuda.is_available():
            batch_input_idx = batch_input_idx.cuda()
            batch_lengths = batch_lengths.cuda()
            batch_word_info = batch_word_info.cuda()
            batch_graph = batch_graph.cuda()

        word_embeds = self.predictor(batch_input_idx, batch_lengths, batch_word_info)
        edge_weight = self.quantizer(word_embeds[batch_graph[:, 1]], word_embeds[batch_graph[:, 2]])
        return self.lossfunc(batch_word_info, batch_graph, edge_weight)
