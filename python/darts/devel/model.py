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
                 max_positions=4096, max_word_positions=32, type_dropout=0.0):
        super().__init__()
        self.type_dropout = float(type_dropout)
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
        # B=1（线上推理/导出）无 padding：跳过 mask 机器，图里无 masked_fill，
        # pnnx 可全原生降级到 ncnn；B>1（训练）保持原 mask 语义不变。
        if batch_input_idx.size(0) == 1 and batch_lengths.size(0) == 1:
            sentence_embedding = self.transformer(vocab_emb)
        else:
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
        relative_positions = (token_positions - word_starts).clamp(0, self.max_word_positions - 1)
        # 池化注意力全程 FP32：word_content_attention 是 Linear(hidden→1) 对整句
        # token 求和，AMP fp16 下梯度动态范围不足，是 skipped_nonfinite_batch
        # 的反复命中点（量产每 epoch 数十次）。该层参数量 hidden×1，FP32 开销
        # 可忽略；softmax/归约同步 FP32，embedding 结果回原 dtype。
        pool_context = sentence_embedding.float()
        content_logits = self.word_content_attention(pool_context).squeeze(-1)[word_batches]
        position_logits = self.word_position_attention(relative_positions).squeeze(-1)
        # 池化 mask 用纯算术窗函数（relu 差）表达：整数格点上严格 0/1，
        # 避免 masked_fill/比较算子（pnnx 无法降级到 ncnn 原生层）。
        tp = token_positions.float()
        lo = (tp - word_starts.float() + 1.0).clamp(0.0, 1.0)
        hi = (word_ends.float() - tp + 1.0).clamp(0.0, 1.0)
        word_gate = lo * hi  # 1 当且仅当 start<=t<=end（整数格点）
        attention_logits = (content_logits + position_logits) + (word_gate - 1.0) * 1e4
        attention = torch.softmax(attention_logits, dim=1)
        # Elementwise reduction is equivalent to a [1, steps] x
        # [steps, hidden] BMM, but avoids unstable cuBLAS BMM backward kernels
        # observed with variable-length span batches on sm_86 GPUs.
        word_embedding = (attention.unsqueeze(-1) * pool_context[word_batches]).sum(dim=1)
        word_lengths = (batch_word_info[:, 2] - batch_word_info[:, 1] + 1).clamp(
            1, self.max_word_positions
        )
        word_embedding = self.word_pool_normal(
            word_embedding + self.word_length_embedding(word_lengths - 1)
        ).to(sentence_embedding.dtype)
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
            word_types = batch_word_info[:, 3]
            if self.training and self.type_dropout > 0:
                # type 列 dropout：随机 UNK 化（PAD 填充与 CLS/SEP 哨兵除外），
                # 防止量化器/识别器把词性当捷径（2026-10-25）
                drop = torch.rand(word_types.shape, device=word_types.device) < self.type_dropout
                keep = (word_types == 0) | (word_types == 2) | (word_types == 3)
                word_types = torch.where(drop & ~keep, torch.full_like(word_types, 1), word_types)
            word_type_embedding = self.type_embedding(word_types)
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
    """Return the negative log-probability of a word association.

    score="cosine"（默认，历史行为）：cosine × 可学习温度，softplus(-logit)。
    score="sigmoid-dot"：NLL = max_nll · sigmoid(dot/τ)，关联 logit = −dot/τ
    （高=关联；DP 与辅助 BCE 桥共用同一 logit 语义）。无除以范数的分母
    （根治零向量梯度爆炸）、NLL 双侧有界 (0, max_nll)、无梯度死区。
    """

    def __init__(self, input_size, hidden_size, distance_buckets=0,
                 score="fused", max_nll=50.0):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.score = score
        self.max_nll = float(max_nll)
        # 可选相对距离感知：相邻候选间隔（跳过的 piece 数）分桶 embedding，
        # 零初始化 → 初始行为与无距离版本逐字节等价（A/B 公平起点）。
        self.distance_buckets = int(distance_buckets)
        self.Kmap = nn.Linear(input_size, hidden_size)
        self.Qmap = nn.Linear(input_size, hidden_size)
        if self.distance_buckets > 0:
            # 注意维度用 input_size（编码器输出维），不是内部投影 hidden
            self.distance_embedding = nn.Embedding(self.distance_buckets, input_size)
            nn.init.zeros_(self.distance_embedding.weight)
        if score in ("sigmoid-dot", "fused"):
            # τ 乘性、初始 1/√d（attention 式 √d 缩放）：raw dot 量级 ~√d，
            # 乘 1/√d 归一至 O(1)，初始 NLL≈5（中性）；可学习并钳制范围
            self.temperature = nn.Parameter(
                torch.tensor(1.0 / np.sqrt(hidden_size), dtype=torch.float32))
        if score == "fused":
            # c 头：softplus 激活（恒正，零初始化基线 ln2≈0.69）——两端词性
            # 强度只有加法贡献，无负负得正死角；dot 加可学习负锚点：
            # 无词性证据时 assoc≈anchor+dot<0 → NLL 大（不成词就要大），
            # 两端皆词时 c1+c2 压过锚点 → NLL 落 0.x（高频主体便宜）。
            # assoc = dot + anchor + c(x1) + c(x2)；NLL = 50·σ(−assoc·τ)
            self.chead = nn.Linear(input_size, 1)
            nn.init.zeros_(self.chead.weight)
            nn.init.zeros_(self.chead.bias)
            self.dot_anchor = nn.Parameter(torch.tensor(0.0))
        else:
            # Cosine similarity is bounded to [-1, 1]. A sqrt(d) initial inverse
            # temperature gives the probability head useful dynamic range early.
            self.logit_scale = nn.Parameter(torch.tensor(np.log(np.sqrt(hidden_size)), dtype=torch.float32))

    def wordness(self, x):
        """c(x)：softplus 恒正的词性强度（仅 fused）。"""
        return F.softplus(self.chead(x)).squeeze(-1)

    def logits(self, x, y, gaps=None):
        """关联 logit（高=更关联）；两条路径语义一致，供 DP/辅助桥共用。"""
        if gaps is not None and self.distance_buckets > 0:
            distance = self.distance_embedding(gaps)
            x = x + distance
            y = y + distance
        if self.score in ("sigmoid-dot", "fused"):
            dot = torch.sum(self.Kmap(x) * self.Qmap(y), dim=-1)
            if self.score == "fused":
                # assoc = dot + anchor − c(x1) − c(x2)；c = softplus 非词度
                # （不成词则大）。−assoc = c1+c2−dot 即拒斥分：任一端非词
                # → c 大 → NLL 大（不成词就要大）；两端皆词 → c≈0 → NLL 由
                # dot 决定（高频主体 0.x）。τ 在 forward 内作用于整个 assoc
                association = (dot + self.dot_anchor
                               - self.wordness(x) - self.wordness(y))
            else:
                # τ ∈ (1e-4, 1.0]：只收紧不放大——sigmoid 饱和区 ±5
                tau = self.temperature.clamp(1e-4, 1.0)
                association = -dot * tau
            return association
        # The default eps=1e-12 makes gradients explode when a projection is
        # close to the zero vector. Such vectors occur naturally early in graph
        # training; a bounded denominator keeps association NLL differentiable.
        keys = F.normalize(self.Kmap(x), dim=-1, eps=1e-4)
        queries = F.normalize(self.Qmap(y), dim=-1, eps=1e-4)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.sum(keys * queries, dim=-1) * scale

    def forward(self, x, y, gaps=None):
        association_logit = self.logits(x, y, gaps)
        if self.score == "fused":
            # NLL = 50·σ(−assoc·τ)——τ 统一缩放 dot 与两个 c 项之和
            tau = self.temperature.clamp(1e-4, 1.0)
            return (self.max_nll * torch.sigmoid(
                -association_logit * tau)).view(-1)
        if self.score == "sigmoid-dot":
            return (self.max_nll * torch.sigmoid(-association_logit)).view(-1)
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
                 pu_target=0.0, pu_weight=0.0, char_negative_weight=1.0,
                 factorized=False):
        super().__init__()
        if class_num < 2:
            raise ValueError("syntax recognizer requires NOT_WORD and at least one POS class")
        self.class_num = class_num
        self.encoder = encoder or WordEncoder(vocab_num, hidden_size, -1)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, class_num))
        # 双头因子化（2026-10-25）：classifier 第 0 列 = 成词 logit w，
        # 1.. 列 = 词上条件标签 logits l。联合 P(NOT_WORD)=σ(−w)、
        # P(k)=σ(w)·softmax(l)_k；损失 = 联合 CE（与旧 BCE(word)+CE(label)
        # 数学同构，仅 w 从聚合 logsumexp 改为专责头）。推理契约双输出：
        # class_probabilities（联合）+ word_probabilities（σ(w)）。
        self.factorized = bool(factorized)
        self.pu_target = pu_target
        self.pu_weight = pu_weight
        # 单字负样本加权：高频单字在其它语境常为词，先验淹没上下文证据
        # （实测 这种车辆 句中 这/种 P≈0.999）；对 len==1 且非词的训练行
        # 放大损失权重，教识别器用上下文压制句内被合并的单字
        self.char_negative_weight = float(char_negative_weight)

    def head_logits(self, batch_input_idx, batch_lengths, batch_span_info):
        """原始头输出：非因子化 = class_num 联合 logits；因子化 = [w | 条件标签 l]。"""
        embeddings = self.encoder(batch_input_idx, batch_lengths, batch_span_info)
        return self.classifier(embeddings)

    def logits(self, batch_input_idx, batch_lengths, batch_span_info):
        """联合 class_num logits（softmax 即联合标签概率），eval/导出统一口径。"""
        raw = self.head_logits(batch_input_idx, batch_lengths, batch_span_info)
        if not self.factorized:
            return raw
        w = raw[:, :1]
        l = raw[:, 1:]
        # 用 sigmoid/softmax/log 基础算子表达（softplus/logsumexp 组合在
        # MNN 转换中降级有数值偏差，2026-10-25 实测 joint 差 0.64）
        word_p = torch.sigmoid(w)
        not_word = torch.log1p(-word_p)
        word_part = torch.log(word_p) + torch.log_softmax(l, dim=-1)
        return torch.cat((not_word, word_part), dim=1)

    def word_logits(self, raw):
        """成词 logit：因子化 = 专责头；旧头 = 聚合 POS 质量对 NOT_WORD。"""
        if self.factorized:
            return raw[:, 0]
        return torch.logsumexp(raw[:, 1:], dim=-1) - raw[:, 0]

    def forward(self, batch_input_idx, batch_lengths, batch_span_info):
        raw = self.head_logits(batch_input_idx, batch_lengths, batch_span_info[:, :3])
        label_column = -2 if batch_span_info.shape[1] > 5 else -1
        labels = batch_span_info[:, label_column]
        is_word = labels != 0
        # 成词项：因子化头 = 专责 w；旧头 = 聚合 POS 质量对 NOT_WORD。
        # 两者都直接训练候选 recall/precision，PU/单字负例加权逻辑共用。
        word_logits = self.word_logits(raw)
        positives = is_word.sum().clamp_min(1)
        negatives = (~is_word).sum().clamp_min(1)
        word_targets = is_word.to(raw.dtype)
        word_weights = torch.ones_like(word_targets)
        if self.pu_weight > 0 and batch_span_info.shape[1] > 5:
            lexicon_unlabelled = batch_span_info[:, -1].bool() & ~is_word
            word_targets = torch.where(
                lexicon_unlabelled, raw.new_tensor(self.pu_target), word_targets
            )
            word_weights = torch.where(
                lexicon_unlabelled, raw.new_tensor(self.pu_weight), word_weights
            )
        word_losses = F.binary_cross_entropy_with_logits(
            word_logits, word_targets,
            pos_weight=(negatives / positives).clamp(max=12.0),
            reduction="none",
        )
        if self.char_negative_weight != 1.0:
            atom_len = batch_span_info[:, 3]
            single_neg = (atom_len == 1) & ~is_word
            word_losses = torch.where(
                single_neg,
                word_losses * self.char_negative_weight, word_losses)
        word_loss = (word_losses * word_weights).sum() / word_weights.sum().clamp_min(1.0)
        if not is_word.any():
            return word_loss
        # 标签项：条件 CE（词上）。因子化头即条件分布；旧头用其 1.. 列同构训练。
        pos_loss = F.cross_entropy(raw[is_word, 1:], labels[is_word] - 1)
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
                probs = torch.softmax(logits, dim=-1)
                if not self.model.factorized:
                    return probs
                raw = self.model.head_logits(token_ids.unsqueeze(0), lengths, batched_spans)
                return probs, torch.sigmoid(raw[:, 0])

        timestep = torch.export.Dim("timestep", min=1)
        spans_count = torch.export.Dim("spans", min=1)
        script = Script(self).eval()
        outputs = ["class_probabilities"]
        axes = {"sents": {0: "timestep"}, "spaninfo": {0: "spans"},
                "class_probabilities": {0: "spans"}}
        if self.factorized:
            outputs.append("word_probabilities")
            axes["word_probabilities"] = {0: "spans"}
        torch.onnx.export(
            script, (sents, spans), outfile, export_params=True, opset_version=18,
            do_constant_folding=True, input_names=["sents", "spaninfo"],
            output_names=outputs,
            dynamic_axes=axes,
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
                 strict_path=True, encoder=None, distance_buckets=0,
                 quantizer_score="cosine"):
        super().__init__()
        self.predictor = encoder or WordEncoder(vocab_num, hidden_size, wtype_num)
        self.quantizer = Quantizer(hidden_size, 32,
                                   distance_buckets=distance_buckets,
                                   score=quantizer_score)
        self.lossfunc = GraphLossSparse(edge_chunk_size=edge_chunk_size, strict_path=strict_path)

    def _edge_gaps(self, batch_word_info, sources, targets):
        """相邻候选间隔（跳过 piece 数）分桶；gap<0 视为 0。"""
        starts = batch_word_info[:, 1]
        ends = batch_word_info[:, 2]
        raw = starts[targets] - ends[sources] - 1
        return raw.clamp(0, self.quantizer.distance_buckets - 1)

    def forward(self, batch_input_idx, batch_lengths, batch_word_info, batch_graph,
                detach_context=False, auxiliary_weight=0.0,
                auxiliary_unlabelled_weight=0.05, auxiliary_max_edges=4096,
                c_weight=0.0, zipf_weight=0.0, atom_ratio_weight=0.0):
        device = next(self.parameters()).device
        batch_input_idx = batch_input_idx.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_word_info = batch_word_info.to(device)
        batch_graph = batch_graph.to(device)

        sources = batch_graph[:, 1]
        targets = batch_graph[:, 2]
        gaps = (self._edge_gaps(batch_word_info, sources, targets)
                if self.quantizer.distance_buckets > 0 else None)
        if detach_context:
            # The exact path objective needs every candidate, but it does not
            # need an autograd graph for the contextual encoder. Build those
            # embeddings without gradients; a bounded node subset below forms
            # the only Transformer bridge.
            with torch.no_grad():
                path_embeds = self.predictor(
                    batch_input_idx, batch_lengths, batch_word_info,
                )
        else:
            path_embeds = self.predictor(
                batch_input_idx, batch_lengths, batch_word_info,
            )
        association_nll = self.quantizer(path_embeds[sources],
                                         path_embeds[targets], gaps)
        path_loss = self.lossfunc(batch_word_info, batch_graph, association_nll)
        total = path_loss
        if (self.quantizer.score == "fused" and c_weight > 0
                and not detach_context):
            # c 头监督：金标路径端点=正（此上下文成词），其余节点=负。
            # c 由上下文嵌入计算 → 学的是"该 span 在该句中是否成词"
            gold_mask = batch_graph[:, 3].bool()
            gold_nodes = torch.zeros(
                batch_word_info.shape[0], device=path_embeds.device)
            gold_nodes[batch_graph[:, 1][gold_mask]] = 1.0
            gold_nodes[batch_graph[:, 2][gold_mask]] = 1.0
            c_logits = self.quantizer.wordness(path_embeds)
            pos = gold_nodes.sum().clamp_min(1.0)
            neg = (1.0 - gold_nodes).sum().clamp_min(1.0)
            total = total + c_weight * F.binary_cross_entropy_with_logits(
                c_logits, gold_nodes, pos_weight=(neg / pos).clamp(max=12.0))
        if zipf_weight > 0:
            # 双端饱和压制（2026-10-24 定稿，替代分位 zipf loss）：σ(·) 在
            # NLL→0 与 NLL→50 两端梯度消失（学习死区）。hinge 逐边惩罚
            # <0.1 与 >49 的边，把权重压回有效动态范围；锚点机制已保证
            # "成词 0.x / 不成词大"的分布形状，这里只防两端堆积
            low = F.relu(0.1 - association_nll)
            high = F.relu(association_nll - 49.0)
            sat = (low.mean() + high.mean()) / self.quantizer.max_nll
            total = total + zipf_weight * sat
        if atom_ratio_weight > 0 and batch_graph.shape[1] > 4:
            # 长度平衡的路径比值损失（2026-10-24 用户定案）：
            #   (gold_sum/√len_gold) / (atom_sum/√len_atom)
            # 消除长短路径 NLL 累加偏置；graph 第 5 列=字符路径边掩码。
            # 梯度同时压金标边、抬字符边；字符链缺失的图自动排除。
            gold_mask_f = batch_graph[:, 3].to(association_nll.dtype)
            atom_mask_f = batch_graph[:, 4].to(association_nll.dtype)
            n_graphs = int(batch_word_info[:, 0].max().item()) + 1
            gold_sum = torch.zeros(n_graphs, dtype=association_nll.dtype,
                                   device=association_nll.device)
            atom_sum = torch.zeros_like(gold_sum)
            gold_cnt = torch.zeros_like(gold_sum)
            atom_cnt = torch.zeros_like(gold_sum)
            gold_sum.index_add_(0, batch_graph[:, 0],
                                association_nll * gold_mask_f)
            atom_sum.index_add_(0, batch_graph[:, 0],
                                association_nll * atom_mask_f)
            gold_cnt.index_add_(0, batch_graph[:, 0], gold_mask_f)
            atom_cnt.index_add_(0, batch_graph[:, 0], atom_mask_f)
            valid = (atom_cnt > 1.5) & (gold_cnt > 1.5) & (atom_sum > 1e-3)
            if valid.any():
                gold_norm = gold_sum[valid] / gold_cnt[valid].sqrt()
                atom_norm = atom_sum[valid] / atom_cnt[valid].sqrt()
                total = total + atom_ratio_weight * (gold_norm / atom_norm).mean()
        if not detach_context or auxiliary_weight <= 0.0:
            return total

        # The dynamic-programming path objective remains responsible for exact
        # graph probabilities, but its long gradient does not enter the
        # Transformer. A local edge objective supplies a short, stable gradient
        # bridge so contextual embeddings still learn association features.
        # Keep every gold transition and deterministically subsample negatives.
        # This truncated edge bridge sends local association information to the
        # Transformer without backpropagating an unbounded candidate DAG.
        gold_indexes = torch.nonzero(batch_graph[:, 3].bool(), as_tuple=False).flatten()
        negative_indexes = torch.nonzero(~batch_graph[:, 3].bool(), as_tuple=False).flatten()
        negative_budget = max(0, int(auxiliary_max_edges) - int(gold_indexes.numel()))
        if negative_budget == 0:
            negative_indexes = negative_indexes[:0]
        elif negative_indexes.numel() > negative_budget:
            positions = torch.linspace(
                0, negative_indexes.numel() - 1, negative_budget,
                device=negative_indexes.device,
            ).long()
            negative_indexes = negative_indexes[positions]
        auxiliary_indexes = torch.cat((gold_indexes, negative_indexes))
        auxiliary_sources = sources[auxiliary_indexes]
        auxiliary_targets = targets[auxiliary_indexes]
        auxiliary_nodes = torch.unique(
            torch.cat((auxiliary_sources, auxiliary_targets)), sorted=True
        )
        node_lookup = torch.full(
            (batch_word_info.shape[0],), -1, dtype=torch.long,
            device=batch_word_info.device,
        )
        node_lookup[auxiliary_nodes] = torch.arange(
            auxiliary_nodes.numel(), device=batch_word_info.device
        )
        auxiliary_embeds = self.predictor(
            batch_input_idx, batch_lengths, batch_word_info[auxiliary_nodes],
        )
        edge_logits = self.quantizer.logits(
            auxiliary_embeds[node_lookup[auxiliary_sources]],
            auxiliary_embeds[node_lookup[auxiliary_targets]],
            (self._edge_gaps(batch_word_info, auxiliary_sources,
                             auxiliary_targets)
             if self.quantizer.distance_buckets > 0 else None),
        )
        edge_targets = batch_graph[auxiliary_indexes, 3].to(edge_logits.dtype)
        edge_losses = F.binary_cross_entropy_with_logits(
            edge_logits, edge_targets, reduction="none"
        )
        edge_weights = torch.where(
            edge_targets.bool(),
            torch.ones_like(edge_targets),
            torch.full_like(edge_targets, auxiliary_unlabelled_weight),
        )
        auxiliary_loss = (edge_losses * edge_weights).sum() / edge_weights.sum().clamp_min(1.0)
        return total + auxiliary_weight * auxiliary_loss


class JointSegmentationTrainer(nn.Module):
    """One shared WordEncoder with a selectable recognizer and graph quantizer."""

    def __init__(self, vocab_num, hidden_size, wtype_num,
                 recognizer_kind="binary", class_num=None, positive_weight=None,
                 pu_target=0.0, pu_weight=0.0, distance_buckets=0,
                 quantizer_score="cosine", char_negative_weight=1.0,
                 type_dropout=0.0, syntax_factorized=False):
        super().__init__()
        encoder = WordEncoder(vocab_num, hidden_size, wtype_num, type_dropout=type_dropout)
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
                char_negative_weight=char_negative_weight,
                factorized=syntax_factorized,
            )
        else:
            raise ValueError(f"unsupported recognizer kind: {recognizer_kind}")
        self.recognizer_kind = recognizer_kind
        self.graph_quantizer = GraphQuantizerTrainer(
            vocab_num, hidden_size, wtype_num, encoder=encoder,
            distance_buckets=distance_buckets,
            quantizer_score=quantizer_score
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
