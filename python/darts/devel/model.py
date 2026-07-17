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


def _finalize_onnx(outfile):
    """Keep dynamo exports loadable by the bundled ONNX Runtime 1.17."""
    import onnx

    model = onnx.load(outfile)
    if model.ir_version > 9:
        model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, outfile)


class WordEncoder(nn.Module):
    """Encode contextual candidate words with content/relative-position pooling.

    ``batch_word_info`` uses ``[batch, first_piece, last_piece, optional_type]``;
    piece bounds are inclusive because this is also the ONNX/C++ contract.  The
    implementation is shared by the recognizer and quantizer as architecture,
    while each task owns and trains a separate parameter set.
    """

    def __init__(self, vocab_num, hidden_size, wtype_num, num_layers=2, num_heads=4,
                 max_positions=4096, max_word_positions=32):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.vocab_num = vocab_num
        self.wtype_num = wtype_num
        self.max_positions = max_positions
        self.max_word_positions = max_word_positions
        self.vocab_embedding = nn.Sequential(
            nn.Embedding(vocab_num, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-7),
            nn.Dropout(0.1),
        )
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # The prototype nested-tensor fast path is unstable for the highly
        # variable padded batches produced by segmentation corpora and has
        # yielded NaN in both efficient-attention and pooling backward kernels.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )
        self.word_content_attention = nn.Linear(hidden_size, 1, bias=False)
        self.word_position_attention = nn.Embedding(max_word_positions, 1)
        self.word_length_embedding = nn.Embedding(max_word_positions, hidden_size)
        self.word_pool_normal = nn.LayerNorm(hidden_size)

        if wtype_num > 0:
            self.type_embedding = nn.Sequential(
                nn.Embedding(wtype_num, hidden_size),
                nn.LayerNorm(hidden_size, eps=1e-7),
                nn.Dropout(0.1),
            )
            self.type_normal = nn.LayerNorm(hidden_size)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info,
                detach_context=False):
        #batch_input_idx (batch*time_step)
        #batch_lengths (batch,)
        #batch_word_info(words_num*[bidx,s,e,tidx])
        steps = batch_input_idx.shape[1]
        positions = torch.arange(steps, device=batch_input_idx.device).clamp_max(self.max_positions - 1)
        vocab_emb = self.vocab_embedding(batch_input_idx) + self.position_embedding(positions).unsqueeze(0)
        padding_mask = positions.unsqueeze(0) >= batch_lengths.unsqueeze(1)
        sentence_embedding = self.transformer(vocab_emb, src_key_padding_mask=padding_mask)
        # Some fused/nested Transformer kernels leave NaN at fully masked
        # padding positions. Zero attention does not protect BMM backward from
        # 0 * NaN, so sanitize padding before any span gather or pooling.
        sentence_embedding = sentence_embedding.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # A candidate is not represented by endpoint averaging.  Each piece in
        # the span receives a content logit and a learned word-relative position
        # bias.  Masked softmax then gives a normalized, length-independent word
        # representation while retaining internal order information.
        word_batches = batch_word_info[:, 0]
        word_starts = batch_word_info[:, 1].unsqueeze(1)
        word_ends = batch_word_info[:, 2].unsqueeze(1)
        token_positions = torch.arange(steps, device=batch_input_idx.device).unsqueeze(0)
        word_mask = (token_positions >= word_starts) & (token_positions <= word_ends)
        relative_positions = (token_positions - word_starts).clamp(0, self.max_word_positions - 1)
        content_logits = self.word_content_attention(sentence_embedding).squeeze(-1)[word_batches]
        position_logits = self.word_position_attention(relative_positions).squeeze(-1)
        attention_logits = (content_logits + position_logits).masked_fill(~word_mask, -1e4)
        attention = torch.softmax(attention_logits, dim=1)
        # Elementwise reduction is equivalent to a [1, steps] x
        # [steps, hidden] BMM, but avoids unstable cuBLAS BMM backward kernels
        # observed with variable-length span batches on sm_86 GPUs.
        word_embedding = (attention.unsqueeze(-1) * sentence_embedding[word_batches]).sum(dim=1)
        word_lengths = (batch_word_info[:, 2] - batch_word_info[:, 1] + 1).clamp(
            1, self.max_word_positions
        )
        word_embedding = self.word_pool_normal(
            word_embedding + self.word_length_embedding(word_lengths - 1)
        )
        if detach_context:
            # Graph path NLL spans many dynamic-programming operations. Let the
            # recognizer train the shared contextual representation and stop
            # that long graph before it reaches Transformer backward. Type
            # embeddings below remain trainable for quantizer specialization.
            word_embedding = word_embedding.detach()
        # In joint training recognizer spans omit the optional type column,
        # while graph nodes include it. The contextual/position parameters are
        # shared in both cases; only graph calls add the type representation.
        if self.wtype_num > 0 and batch_word_info.shape[1] > 3:
            word_type_embedding = self.type_embedding(batch_word_info[:, 3])
            return self.type_normal(word_embedding + word_type_embedding)
        return word_embedding

    def export2onnx(self, outfile="transformer.encoder.onnx"):
        sents_idx = torch.randint(0, self.vocab_num, (11, ))
        word_se = torch.LongTensor([[0, 0], [1, 2], [3, 3], [4, 6], [7, 7], [8, 9], [10, 10]])
        wtype_idx = torch.randint(0, self.wtype_num, (word_se.shape[0], 1))
        word_info = torch.concat((word_se, wtype_idx), dim=1)

        class _script(nn.Module):

            def __init__(self, obj) -> None:
                super().__init__()
                self.obj = obj

            def forward(self, sents, words):
                lens = torch.ones((1,), dtype=torch.long, device=sents.device) * sents.shape[0]
                bsents = torch.unsqueeze(sents, 0)
                bidx = torch.zeros((words.shape[0], 1), dtype=wtype_idx.dtype)
                words = torch.concat((bidx, words), dim=1)
                return self.obj(bsents, lens, words)

        # Export one sentence; the C++ indicator supplies the same tensor layout.
        inputdata = (sents_idx, word_info)
        inputnames = ['sents', 'wordinfo']
        dynamic_axes = {"sents": {0: 'timestep'}, "wordinfo": {0: 'wordnums'}, "wordemb": {0: 'wordnums'}}
        timestep = torch.export.Dim("timestep", min=1)
        wordnums = torch.export.Dim("wordnums", min=1)

        script = _script(self).eval()
        torch.onnx.export(
            script,
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=18,  # Dynamo's native opset avoids lossy version conversion.
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['wordemb'],  # the model's output names
            dynamic_axes=dynamic_axes,
            dynamic_shapes=({0: timestep}, {0: wordnums}),
            external_data=False,
            dynamo=True)
        _finalize_onnx(outfile)
        return outfile


class Quantizer(nn.Module):
    """Return the negative log-probability of a word association."""

    def __init__(self, input_size, hidden_size):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kmap = nn.Linear(input_size, hidden_size)
        self.Qmap = nn.Linear(input_size, hidden_size)
        # Cosine similarity is bounded to [-1, 1]. A sqrt(d) initial inverse
        # temperature gives the probability head useful dynamic range early.
        self.logit_scale = nn.Parameter(torch.tensor(np.log(np.sqrt(hidden_size)), dtype=torch.float32))

    def logits(self, x, y):
        # The default eps=1e-12 makes gradients explode when a projection is
        # close to the zero vector. Such vectors occur naturally early in graph
        # training; a bounded denominator keeps association NLL differentiable.
        keys = F.normalize(self.Kmap(x), dim=-1, eps=1e-4)
        queries = F.normalize(self.Qmap(y), dim=-1, eps=1e-4)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.sum(keys * queries, dim=-1) * scale

    def forward(self, x, y):
        association_logit = self.logits(x, y)
        # softplus(-x) is the stable form of -log(sigmoid(x)).
        return F.softplus(-association_logit).view(-1)

    def export2onnx(self, outfile="sample.quantizer.onnx"):
        inputdata = (torch.randn((1, self.input_size)), torch.randn((1, self.input_size)))
        inputnames = ['a', 'b']
        dynamic_axes = {'a': {0: 'edges'}, 'b': {0: 'edges'}, 'association_nll': {0: 'edges'}}
        edges = torch.export.Dim("edges", min=1)

        torch.onnx.export(
            self,
            inputdata,
            outfile,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=18,
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=inputnames,  # the model's input names
            output_names=['association_nll'],
            dynamic_axes=dynamic_axes,
            dynamic_shapes=({0: edges}, {0: edges}),
            external_data=False,
            dynamo=True)
        _finalize_onnx(outfile)
        return outfile


class SpanRecognizer(nn.Module):
    """Predict independent word probabilities for overlapping candidate spans.

    Training rows are ``[batch, first_piece, last_piece, atom_length, label]``.
    Atom length is metadata for threshold calibration and is intentionally not a
    model feature, so Python training and C++ inference use the same three-column
    span tensor.
    """

    def __init__(self, vocab_num, hidden_size, encoder=None, positive_weight=None,
                 pu_target=0.0, pu_weight=0.0):
        super().__init__()
        self.encoder = encoder or WordEncoder(vocab_num, hidden_size, -1)
        self.probability_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 1))
        self.positive_weight = positive_weight
        self.pu_target = pu_target
        self.pu_weight = pu_weight

    def logits(self, batch_input_idx, batch_lengths, batch_span_info):
        embeddings = self.encoder(batch_input_idx, batch_lengths, batch_span_info)
        return self.probability_head(embeddings).squeeze(-1)

    def forward(self, batch_input_idx, batch_lengths, batch_span_info):
        logits = self.logits(batch_input_idx, batch_lengths, batch_span_info[:, :3])
        # Evaluation readers append a gold-lexicon validity column after the
        # annotated label. Legacy/manual five-column tensors remain supported.
        label_column = -2 if batch_span_info.shape[1] > 5 else -1
        labels = batch_span_info[:, label_column].to(logits.dtype)
        targets = labels
        weights = torch.ones_like(labels)
        if self.pu_weight > 0 and batch_span_info.shape[1] > 5:
            lexicon_unlabelled = batch_span_info[:, -1].bool() & ~labels.bool()
            targets = torch.where(
                lexicon_unlabelled, logits.new_tensor(self.pu_target), labels
            )
            weights = torch.where(
                lexicon_unlabelled, logits.new_tensor(self.pu_weight), weights
            )
        if self.positive_weight is None:
            positives = labels.sum().clamp_min(1.0)
            negatives = (1.0 - labels).sum().clamp_min(1.0)
            positive_weight = (negatives / positives).clamp(max=20.0)
        else:
            # A corpus-level prior keeps the logit scale identical across
            # batches. Per-batch weighting makes exported probabilities depend
            # on the accidental class mixture of every optimizer step.
            positive_weight = logits.new_tensor(self.positive_weight)
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=positive_weight, reduction="none"
        )
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)

    def export2onnx(self, outfile="span.recognizer.onnx"):
        sents = torch.randint(0, self.encoder.vocab_num, (11,))
        spans = torch.LongTensor([[0, 1], [1, 3], [3, 6], [6, 10]])

        class Script(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, token_ids, span_info):
                lengths = torch.ones((1,), dtype=torch.long, device=token_ids.device) * token_ids.shape[0]
                batch_ids = torch.zeros((span_info.shape[0], 1), dtype=torch.long, device=span_info.device)
                batched_spans = torch.cat((batch_ids, span_info), dim=1)
                logits = self.model.logits(token_ids.unsqueeze(0), lengths, batched_spans)
                return torch.sigmoid(logits)

        timestep = torch.export.Dim("timestep", min=1)
        spans_count = torch.export.Dim("spans", min=1)

        script = Script(self).eval()
        torch.onnx.export(
            script, (sents, spans), outfile, export_params=True, opset_version=18,
            do_constant_folding=True, input_names=["sents", "spaninfo"],
            output_names=["word_probabilities"],
            dynamic_axes={"sents": {0: "timestep"}, "spaninfo": {0: "spans"},
                          "word_probabilities": {0: "spans"}},
            dynamic_shapes=({0: timestep}, {0: spans_count}),
            external_data=False,
            dynamo=True,
        )
        _finalize_onnx(outfile)
        return outfile


class SyntaxSpanRecognizer(nn.Module):
    """Classify every span as NOT_WORD or one mutually exclusive POS type."""

    def __init__(self, vocab_num, hidden_size, class_num, encoder=None,
                 pu_target=0.0, pu_weight=0.0):
        super().__init__()
        if class_num < 2:
            raise ValueError("syntax recognizer requires NOT_WORD and at least one POS class")
        self.class_num = class_num
        self.encoder = encoder or WordEncoder(vocab_num, hidden_size, -1)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, class_num))
        self.pu_target = pu_target
        self.pu_weight = pu_weight

    def logits(self, batch_input_idx, batch_lengths, batch_span_info):
        embeddings = self.encoder(batch_input_idx, batch_lengths, batch_span_info)
        return self.classifier(embeddings)

    def forward(self, batch_input_idx, batch_lengths, batch_span_info):
        logits = self.logits(batch_input_idx, batch_lengths, batch_span_info[:, :3])
        label_column = -2 if batch_span_info.shape[1] > 5 else -1
        labels = batch_span_info[:, label_column]
        is_word = labels != 0
        # Compare the aggregate POS mass against NOT_WORD. This directly trains
        # candidate recall/precision without letting sparse POS reweighting
        # suppress the abundant and important NOT_WORD class.
        word_logits = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        positives = is_word.sum().clamp_min(1)
        negatives = (~is_word).sum().clamp_min(1)
        word_targets = is_word.to(logits.dtype)
        word_weights = torch.ones_like(word_targets)
        if self.pu_weight > 0 and batch_span_info.shape[1] > 5:
            lexicon_unlabelled = batch_span_info[:, -1].bool() & ~is_word
            word_targets = torch.where(
                lexicon_unlabelled, logits.new_tensor(self.pu_target), word_targets
            )
            word_weights = torch.where(
                lexicon_unlabelled, logits.new_tensor(self.pu_weight), word_weights
            )
        word_losses = F.binary_cross_entropy_with_logits(
            word_logits, word_targets,
            pos_weight=(negatives / positives).clamp(max=12.0),
            reduction="none",
        )
        word_loss = (word_losses * word_weights).sum() / word_weights.sum().clamp_min(1.0)
        if not is_word.any():
            return word_loss
        # POS classification is conditioned on known word spans, so it cannot
        # trade word recall against rare-class weights.
        pos_loss = F.cross_entropy(logits[is_word, 1:], labels[is_word] - 1)
        return word_loss + pos_loss

    def export2onnx(self, outfile="syntax.recognizer.onnx"):
        sents = torch.randint(0, self.encoder.vocab_num, (11,))
        spans = torch.LongTensor([[0, 0], [0, 1], [2, 4], [5, 10]])

        class Script(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, token_ids, span_info):
                lengths = torch.ones((1,), dtype=torch.long, device=token_ids.device) * token_ids.shape[0]
                batch_ids = torch.zeros((span_info.shape[0], 1), dtype=torch.long,
                                        device=span_info.device)
                batched_spans = torch.cat((batch_ids, span_info), dim=1)
                logits = self.model.logits(token_ids.unsqueeze(0), lengths, batched_spans)
                return torch.softmax(logits, dim=-1)

        timestep = torch.export.Dim("timestep", min=1)
        spans_count = torch.export.Dim("spans", min=1)
        script = Script(self).eval()
        torch.onnx.export(
            script, (sents, spans), outfile, export_params=True, opset_version=18,
            do_constant_folding=True, input_names=["sents", "spaninfo"],
            output_names=["class_probabilities"],
            dynamic_axes={"sents": {0: "timestep"}, "spaninfo": {0: "spans"},
                          "class_probabilities": {0: "spans"}},
            dynamic_shapes=({0: timestep}, {0: spans_count}),
            external_data=False, dynamo=True,
        )
        _finalize_onnx(outfile)
        return outfile


class GraphQuantizerTrainer(nn.Module):
    """
    Sparse batched graph trainer.

    The recognizer model is not reused here: only the WordEncoder implementation
    is shared.  This model is independently optimized and exported as indicator
    plus quantizer ONNX files.

    batch_graph is expected to be a flat edge table:
    [batch_id, src_node, dst_node, gold_mask]
    where src/dst are flattened node ids aligned with batch_word_info.
    """

    def __init__(self, vocab_num, hidden_size, wtype_num, edge_chunk_size=65536,
                 strict_path=True, encoder=None):
        super().__init__()
        self.predictor = encoder or WordEncoder(vocab_num, hidden_size, wtype_num)
        self.quantizer = Quantizer(hidden_size, 32)
        self.lossfunc = GraphLossSparse(edge_chunk_size=edge_chunk_size, strict_path=strict_path)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info, batch_graph,
                detach_context=False, auxiliary_weight=0.0,
                auxiliary_unlabelled_weight=0.05):
        device = next(self.parameters()).device
        batch_input_idx = batch_input_idx.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_word_info = batch_word_info.to(device)
        batch_graph = batch_graph.to(device)

        word_embeds = self.predictor(
            batch_input_idx, batch_lengths, batch_word_info
        )
        sources = batch_graph[:, 1]
        targets = batch_graph[:, 2]
        path_embeds = word_embeds.detach() if detach_context else word_embeds
        association_nll = self.quantizer(path_embeds[sources], path_embeds[targets])
        path_loss = self.lossfunc(batch_word_info, batch_graph, association_nll)
        if not detach_context or auxiliary_weight <= 0.0:
            return path_loss

        # The dynamic-programming path objective remains responsible for exact
        # graph probabilities, but its long gradient does not enter the
        # Transformer. A local edge objective supplies a short, stable gradient
        # bridge so contextual embeddings still learn association features.
        edge_logits = self.quantizer.logits(word_embeds[sources], word_embeds[targets])
        edge_targets = batch_graph[:, 3].to(edge_logits.dtype)
        edge_losses = F.binary_cross_entropy_with_logits(
            edge_logits, edge_targets, reduction="none"
        )
        edge_weights = torch.where(
            edge_targets.bool(),
            torch.ones_like(edge_targets),
            torch.full_like(edge_targets, auxiliary_unlabelled_weight),
        )
        auxiliary_loss = (edge_losses * edge_weights).sum() / edge_weights.sum().clamp_min(1.0)
        return path_loss + auxiliary_weight * auxiliary_loss


class JointSegmentationTrainer(nn.Module):
    """One shared WordEncoder with a selectable recognizer and graph quantizer."""

    def __init__(self, vocab_num, hidden_size, wtype_num,
                 recognizer_kind="binary", class_num=None, positive_weight=None,
                 pu_target=0.0, pu_weight=0.0):
        super().__init__()
        encoder = WordEncoder(vocab_num, hidden_size, wtype_num)
        if recognizer_kind == "binary":
            self.recognizer = SpanRecognizer(
                vocab_num, hidden_size, encoder=encoder, positive_weight=positive_weight,
                pu_target=pu_target, pu_weight=pu_weight,
            )
        elif recognizer_kind == "syntax":
            if class_num is None:
                raise ValueError("syntax joint training requires class_num")
            self.recognizer = SyntaxSpanRecognizer(
                vocab_num, hidden_size, class_num, encoder=encoder,
                pu_target=pu_target, pu_weight=pu_weight,
            )
        else:
            raise ValueError(f"unsupported recognizer kind: {recognizer_kind}")
        self.recognizer_kind = recognizer_kind
        self.graph_quantizer = GraphQuantizerTrainer(
            vocab_num, hidden_size, wtype_num, encoder=encoder
        )

    @property
    def encoder(self):
        return self.recognizer.encoder

    def set_training_stage(self, stage):
        """Select trainable parameters without breaking the shared encoder."""
        if stage not in ("joint", "recognizer", "quantizer"):
            raise ValueError(f"unsupported training stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "joint")
        if stage == "recognizer":
            for parameter in self.recognizer.parameters():
                parameter.requires_grad_(True)
        elif stage == "quantizer":
            # Stage two consumes the fixed recognizer representation and only
            # adapts the association head plus type-only features. Recognizer
            # spans omit the type column, so these parameters cannot alter its
            # probabilities but must not remain at random initialization.
            for parameter in self.graph_quantizer.quantizer.parameters():
                parameter.requires_grad_(True)
            for parameter in self.encoder.type_embedding.parameters():
                parameter.requires_grad_(True)
            for parameter in self.encoder.type_normal.parameters():
                parameter.requires_grad_(True)
