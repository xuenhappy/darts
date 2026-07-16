'''
File: reader.py
Project: devel
File Created: Saturday, 8th January 2022 9:12:42 pm
Author: Xu En (nanhangxuen@163.com)
-----
Last Modified: Saturday, 8th January 2022 9:13:03 pm
Modified By: Xu En (nanhangxuen@163.com)
-----
Copyright 2021 - 2022 XuEn
'''
import torch
import numpy as np
import random
from ..cdarts import *
from torch.utils.data import IterableDataset


def d2list2array(d2list, fval=0, dtype=np.int32):
    """
    convert 2d length no eq array  arralist to a 2d matrix
    """
    lens = np.asarray(list(len(item) for item in d2list), np.int32)
    mask = lens[:, None] > np.arange(lens.max())
    out = np.ones(mask.shape, dtype=dtype) * fval
    out[mask] = np.concatenate(d2list)
    return out, lens


def piece_bounds(codes, atom_count):
    """Map each Atom to its inclusive WordPiece bounds."""
    starts = [-1] * atom_count
    ends = [-1] * atom_count
    for piece_index, (_code, atom_position) in enumerate(codes):
        if atom_position < 0:
            continue
        if atom_position >= atom_count:
            raise ValueError(f"WordPiece atom position {atom_position} exceeds {atom_count} atoms")
        if starts[atom_position] < 0:
            starts[atom_position] = piece_index
        ends[atom_position] = piece_index
    missing = [index for index, start in enumerate(starts) if start < 0]
    if missing:
        raise ValueError(f"WordPiece encoder produced no pieces for atoms {missing}")
    return starts, ends


class GraphSampleReader(IterableDataset):
    """Build candidate DAGs aligned with the C++ ONNX decider contract.

    Nodes are sorted candidate words plus synthetic head/tail nodes.  Edges join
    exactly adjacent half-open atom spans, and gold_mask marks one complete path.
    Flattened node indexes allow GraphLossSparse to train a batch without dense
    ``V x V`` allocation.
    """

    def __init__(self, sample, config="data/conf.json", mode="hybrid", batch_size=16,
                 max_span=5, shuffle=False, type_map=None):
        super().__init__()
        self.sample = sample
        self.batch_size = batch_size
        self.max_span = max_span
        self.shuffle = shuffle
        self._samples = None
        self.segment = DSegment(config, mode, isdev=True)
        self.atom_codec = AtomCodec({"base.dir": "data/models/codex"})
        if type_map is None:
            type_map = "data/codes/pos.hx.txt" if mode == "lac" else "data/codes/type.hx.txt"
        self.word_codec = WordCodec({"hx.file": type_map}, "LabelEncoder")
        self.type_codes = {
            self.word_codec.decode(code): code for code in range(self.word_codec.label_nums())
        }

    def wordsize(self):
        return self.atom_codec.label_nums()

    def typesize(self):
        return self.word_codec.label_nums()

    @staticmethod
    def _gold_spans(tokens, atoms):
        """Project corpus token boundaries onto the sentence AtomList.

        Atomization must happen after tokens are concatenated because adjacent
        ENG or NUM tokens may become one runtime Atom. A boundary inside such an
        Atom cannot be represented by ``best_path`` and is therefore merged
        with the next gold token. Returned indexes always address ``atoms``.
        """
        atom_values = atoms.tolist()
        if not atom_values:
            return []
        # Readers rebuild a canonical single-space segmented sentence. Spaces
        # are not Atoms, but they keep adjacent ENG/NUM gold words from being
        # merged by the atomizer.
        text_length = sum(len(token) for token in tokens) + len(tokens) - 1
        if atom_values[0].st != 0 or atom_values[-1].et != text_length:
            raise ValueError("gold text offsets do not cover the sentence AtomList")
        atom_boundaries = {0: 0}
        atom_boundaries.update({atom.et: index + 1 for index, atom in enumerate(atom_values)})
        boundaries = [0]
        text_offset = 0
        for token in tokens[:-1]:
            text_offset += len(token)
            atom_offset = atom_boundaries.get(text_offset)
            if atom_offset is None:
                raise ValueError(f"gold boundary at text offset {text_offset} is not an Atom boundary")
            if atom_offset != boundaries[-1]:
                boundaries.append(atom_offset)
            text_offset += 1
        boundaries.append(len(atom_values))
        return list(zip(boundaries, boundaries[1:]))

    @staticmethod
    def _tagged_tokens(items):
        words = []
        tags = []
        for item in items:
            word, separator, tag = item.rpartition("/")
            if separator and word and tag.startswith("POS_"):
                words.append(word)
                tags.append(tag)
            else:
                words.append(item)
                tags.append(None)
        return words, tags

    def _sample(self, items):
        tokens, gold_tags = self._tagged_tokens(items)
        # Spaces separate adjacent ENG/NUM words during atomization, then
        # skip_space removes them from the Atom index space.
        text = " ".join(tokens)
        atoms, candidates = self.segment.cut(
            text, max_mode=True, skip_space=True, normal_before=False
        )
        words = candidates.tolist()
        types = self.word_codec.encode(candidates)
        codes = self.atom_codec.encode(atoms)
        code_ids = [value[0] for value in codes]
        starts, ends = piece_bounds(codes, len(atoms))

        # Dictionary labels provide known type features.  Add the complete span
        # envelope used by the recognizer so the quantizer also sees OOV words;
        # type 0 is the LabelEncoder's unknown fallback.
        candidates_by_node = {}
        for word, type_code in zip(words, types):
            candidates_by_node[(word.atom_s, word.atom_e, type_code)] = None
        for start in range(len(atoms)):
            for end in range(start + 1, min(len(atoms), start + self.max_span) + 1):
                candidates_by_node.setdefault((start, end, 0), None)
        gold_spans = self._gold_spans(tokens, atoms)
        gold_nodes_by_span = []
        for span, tag in zip(gold_spans, gold_tags):
            type_code = self.type_codes.get(tag, 0) if tag else 0
            node_key = (*span, type_code)
            candidates_by_node.setdefault(node_key, None)
            gold_nodes_by_span.append(node_key)

        nodes = [(0, 0, 3, -1, 0)]
        for atom_start, atom_end, type_code in sorted(candidates_by_node):
            nodes.append((starts[atom_start], ends[atom_end - 1], type_code, atom_start, atom_end))
        nodes.append((len(code_ids) - 1, len(code_ids) - 1, 2, len(atoms), len(atoms) + 1))

        gold_nodes = [0]
        for atom_start, atom_end, type_code in gold_nodes_by_span:
            match = next((index for index, node in enumerate(nodes)
                          if node[2] == type_code and node[3:5] == (atom_start, atom_end)), None)
            if match is None:
                raise RuntimeError(
                    f"gold node {(atom_start, atom_end, type_code)} missing from graph for {text}"
                )
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
        # Keep this contract identical to GraphSampleReader.
        text = " ".join(tokens)
        atoms = PyAtomList(text, skip_space=True, normal_before=False)
        codes = self.atom_codec.encode(atoms)
        code_ids = [item[0] for item in codes]
        starts, ends = piece_bounds(codes, len(atoms))
        gold = set(GraphSampleReader._gold_spans(tokens, atoms))
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


class SyntaxSpanSampleReader(IterableDataset):
    """Generate span classes where 0 is NOT_WORD and 1..N are POS labels."""

    def __init__(self, sample, type_map="data/codes/pos.hx.txt", batch_size=32,
                 max_span=5, shuffle=False):
        super().__init__()
        self.sample = sample
        self.batch_size = batch_size
        self.max_span = max_span
        self.shuffle = shuffle
        self._samples = None
        self.atom_codec = AtomCodec({"base.dir": "data/models/codex"})
        self.labels = ["NOT_WORD"]
        with open(type_map, encoding="utf-8") as stream:
            self.labels.extend(line.split("#", 1)[0].strip() for line in stream if "#" in line)
        self.label_codes = {label: index for index, label in enumerate(self.labels)}

    def wordsize(self):
        return self.atom_codec.label_nums()

    def classsize(self):
        return len(self.labels)

    def _sample(self, items):
        tokens, tags = GraphSampleReader._tagged_tokens(items)
        text = " ".join(tokens)
        atoms = PyAtomList(text, skip_space=True, normal_before=False)
        codes = self.atom_codec.encode(atoms)
        code_ids = [item[0] for item in codes]
        starts, ends = piece_bounds(codes, len(atoms))
        gold = {
            span: self.label_codes.get(tag, 0)
            for span, tag in zip(GraphSampleReader._gold_spans(tokens, atoms), tags)
        }
        spans = []
        for start in range(len(atoms)):
            for end in range(start + 1, min(len(atoms), start + self.max_span) + 1):
                spans.append((starts[start], ends[end - 1], end - start,
                              gold.get((start, end), 0)))
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
                self._samples = [
                    self._sample(line.strip().split()) for line in stream if line.strip()
                ]
        ordered = list(self._samples)
        if self.shuffle:
            random.shuffle(ordered)
        for start in range(0, len(ordered), self.batch_size):
            yield self._batch(ordered[start:start + self.batch_size])
