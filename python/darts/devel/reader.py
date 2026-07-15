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
import random
from ..cdarts import *
from torch.utils.data import Dataset, DataLoader, IterableDataset


def d2list2array(d2list, fval=0, dtype=np.int32):
    """
    convert 2d length no eq array  arralist to a 2d matrix
    """
    lens = np.asarray(list(len(item) for item in d2list), np.int32)
    mask = lens[:, None] > np.arange(lens.max())
    out = np.ones(mask.shape, dtype=dtype) * fval
    out[mask] = np.concatenate(d2list)
    return out, lens


class GraphSampleReader(IterableDataset):
    """Build candidate DAGs aligned with the C++ ONNX decider contract.

    Nodes are sorted candidate words plus synthetic head/tail nodes.  Edges join
    exactly adjacent half-open atom spans, and gold_mask marks one complete path.
    Flattened node indexes allow GraphLossSparse to train a batch without dense
    ``V x V`` allocation.
    """

    def __init__(self, sample, config="data/conf.json", mode="hybrid", batch_size=16,
                 max_span=5, shuffle=False):
        super().__init__()
        self.sample = sample
        self.batch_size = batch_size
        self.max_span = max_span
        self.shuffle = shuffle
        self._samples = None
        self.segment = DSegment(config, mode, isdev=True)
        self.atom_codec = AtomCodec({"base.dir": "data/models/codex"})
        self.word_codec = WordCodec({"hx.file": "data/codes/type.hx.txt"}, "LabelEncoder")

    def wordsize(self):
        return self.atom_codec.label_nums()

    def typesize(self):
        return self.word_codec.label_nums()

    @staticmethod
    def _gold_spans(tokens):
        spans = []
        position = 0
        for token in tokens:
            length = len(PyAtomList(token, skip_space=True, normal_before=False))
            spans.append((position, position + length))
            position += length
        return spans

    def _sample(self, tokens):
        text = "".join(tokens)
        atoms, candidates = self.segment.cut(text, max_mode=True)
        words = candidates.tolist()
        types = self.word_codec.encode(candidates)
        codes = self.atom_codec.encode(atoms)
        code_ids = [value[0] for value in codes]
        starts = [0] * len(atoms)
        ends = [0] * len(atoms)
        for index, (_code, atom_position) in enumerate(codes):
            if atom_position < 0:
                continue
            if starts[atom_position] == 0:
                starts[atom_position] = index
            ends[atom_position] = index

        # Dictionary labels provide known type features.  Add the complete span
        # envelope used by the recognizer so the quantizer also sees OOV words;
        # type 0 is the LabelEncoder's unknown fallback.
        candidates_by_span = {}
        for word, type_code in zip(words, types):
            candidates_by_span.setdefault((word.atom_s, word.atom_e), type_code)
        for start in range(len(atoms)):
            for end in range(start + 1, min(len(atoms), start + self.max_span) + 1):
                candidates_by_span.setdefault((start, end), 0)
        gold_spans = self._gold_spans(tokens)
        for span in gold_spans:
            candidates_by_span.setdefault(span, 0)

        nodes = [(0, 0, 3, -1, 0)]
        for (atom_start, atom_end), type_code in sorted(candidates_by_span.items()):
            nodes.append((starts[atom_start], ends[atom_end - 1], type_code, atom_start, atom_end))
        nodes.append((len(code_ids) - 1, len(code_ids) - 1, 2, len(atoms), len(atoms) + 1))

        gold_nodes = [0]
        for span in gold_spans:
            match = next((index for index, node in enumerate(nodes) if node[3:5] == span), None)
            if match is None:
                raise RuntimeError(f"gold span {span} missing from candidate graph for {text}")
            gold_nodes.append(match)
        gold_nodes.append(len(nodes) - 1)
        gold_edges = set(zip(gold_nodes, gold_nodes[1:]))

        edges = []
        for source, left in enumerate(nodes[:-1]):
            target_start = 0 if source == 0 else left[4]
            for target in range(1, len(nodes)):
                right = nodes[target]
                if right[3] == target_start:
                    edges.append((source, target, int((source, target) in gold_edges)))
        return code_ids, nodes, edges

    def _batch(self, samples):
        code_ids, lengths = d2list2array([sample[0] for sample in samples])
        word_info = []
        graph = []
        offset = 0
        for batch_id, (_codes, nodes, edges) in enumerate(samples):
            word_info.extend((batch_id, node[0], node[1], node[2]) for node in nodes)
            graph.extend((batch_id, source + offset, target + offset, gold)
                         for source, target, gold in edges)
            offset += len(nodes)
        graph.sort(key=lambda edge: (edge[0], edge[2], edge[1]))
        return (torch.from_numpy(code_ids).long(), torch.from_numpy(lengths).long(),
                torch.tensor(word_info, dtype=torch.long), torch.tensor(graph, dtype=torch.long))

    def __iter__(self):
        if self._samples is None:
            with open(self.sample, encoding="utf-8") as stream:
                lines = list(stream)
            self._samples = [self._sample(line.strip().split()) for line in lines if line.strip()]
        ordered = list(self._samples)
        if self.shuffle:
            random.shuffle(ordered)
        for start in range(0, len(ordered), self.batch_size):
            yield self._batch(ordered[start:start + self.batch_size])


class SpanSampleReader(IterableDataset):
    """Generate all overlapping 2..max_span candidates and binary labels.

    Gold segmentation is used only to label independently possible words; no BIO
    path is created.  Atom bounds are half-open here, then converted to inclusive
    WordPiece bounds for WordEncoder and the exported ONNX model.
    """

    def __init__(self, sample, batch_size=32, max_span=5, shuffle=False):
        super().__init__()
        self.sample = sample
        self.batch_size = batch_size
        self.max_span = max_span
        self.shuffle = shuffle
        self._samples = None
        self.atom_codec = AtomCodec({"base.dir": "data/models/codex"})

    def wordsize(self):
        return self.atom_codec.label_nums()

    def _sample(self, tokens):
        text = "".join(tokens)
        atoms = PyAtomList(text)
        codes = self.atom_codec.encode(atoms)
        code_ids = [item[0] for item in codes]
        starts = [0] * len(atoms)
        ends = [0] * len(atoms)
        for index, (_code, atom_position) in enumerate(codes):
            if atom_position < 0:
                continue
            if starts[atom_position] == 0:
                starts[atom_position] = index
            ends[atom_position] = index
        gold = set(GraphSampleReader._gold_spans(tokens))
        spans = []
        for start in range(len(atoms)):
            for end in range(start + 2, min(len(atoms), start + self.max_span) + 1):
                spans.append((starts[start], ends[end - 1], end - start, int((start, end) in gold)))
        return code_ids, spans

    def _batch(self, samples):
        code_ids, lengths = d2list2array([sample[0] for sample in samples])
        spans = [(batch_id, start, end, atom_length, label)
                 for batch_id, (_codes, sample_spans) in enumerate(samples)
                 for start, end, atom_length, label in sample_spans]
        return (torch.from_numpy(code_ids).long(), torch.from_numpy(lengths).long(),
                torch.tensor(spans, dtype=torch.long))

    def __iter__(self):
        if self._samples is None:
            with open(self.sample, encoding="utf-8") as stream:
                lines = list(stream)
            self._samples = []
            for line in lines:
                tokens = line.strip().split()
                if not tokens:
                    continue
                sample = self._sample(tokens)
                if sample[1]:
                    self._samples.append(sample)
        ordered = list(self._samples)
        if self.shuffle:
            random.shuffle(ordered)
        for start in range(0, len(ordered), self.batch_size):
            yield self._batch(ordered[start:start + self.batch_size])
