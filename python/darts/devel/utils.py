'''
File: torch_tools.py
Project: devel
File Created: Saturday, 8th January 2022 9:07:18 pm
Author: Xu En (xuen@mokar.com)
-----
Last Modified: Saturday, 8th January 2022 9:07:31 pm
Modified By: Xu En (xuen@mokahr.com)
-----
Copyright 2021 - 2022 Your Company, Moka
'''
import numpy as np
import torch
from torch_scatter import segment_max_coo
import torch.nn as nn


def pin2cuda(*args):
    return [t.pin_memory().cuda(non_blocking=True) for t in args]


def swp2cuda(*args):
    return [t.cuda() for t in args]


def freeze_module(model, keep_back=True):
    """
    freeze model to use as a fixed function
    """
    model.eval()
    for param in model.parameters():
        param.drop_grad = True
        param.requires_grad = keep_back


def drop_module_grad(model):
    """
    if you call freeze_module() to set model grad None to free it
    this method must call after loss.backward() and  before optimizer.step()
    """
    # nn.utils.clip_grad_norm_(model.parameters(), 1, 'inf')
    # for name, p in model.named_parameters():
    #     print(name, p.grad, p.shape)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if hasattr(p, 'drop_grad') and p.drop_grad:
            p.grad = None


def unfreeze_module(model):
    """
    call this model
    """
    model.train(True)
    for param in model.parameters():
        if hasattr(param, 'drop_grad'):
            delattr(param, 'drop_grad')
            param.requires_grad = True


def run_rnn(rnn, inputs, seq_lengths):
    """
    run the rnn 
    the input shape must (N,T,C)
    seq_lengths shape is (N,)
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    _, desorted_indices = torch.sort(indices, descending=False)
    inputs = inputs.index_select(0, indices)
    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths.cpu(), batch_first=True)
    rnn.flatten_parameters()
    res, _ = rnn(packed_inputs)
    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True, total_length=inputs.shape[1])
    return padded_res.index_select(0, desorted_indices).contiguous()


def batch_segment_max(arr, batch_slices):
    """arr is a 2d or 3d tensor and shape is (N,T,C),
    batch_slices is a long tensor and shape is (M,3),
    the meaning of batch_slices's colum is [batch idx,start idx,end idx]
    """
    def getGroupMask(batch_slice):
        tmpMatLen = batch_slice[:, 2]-batch_slice[:, 1]+1
        clens = torch.cumsum(tmpMatLen, 0)
        idxs = torch.arange(0, torch.sum(tmpMatLen), device=clens.device)
        group = torch.searchsorted(clens, idxs, right=True)
        inc = clens[group]-idxs-1
        ridx = batch_slice[:, 0][group]
        cidx = batch_slice[:, 2][group]-inc
        return group, ridx, cidx
    groups, ridx, cidx = getGroupMask(batch_slices)
    return segment_max_coo(arr[ridx, cidx], groups)[0]


def batch_segment_fill(arr, batch_slices, val):
    """arr is a 2d or 3d tensor and shape is (N,T),
    batch_slices is a long tensor and shape is (M,3),
    the meaning of batch_slices's colum is [batch idx,start idx,end idx]
    val is a tensor that shape is (M,)
    """
    def getGroupMask(batch_slice):
        tmpMatLen = batch_slice[:, 2]-batch_slice[:, 1]+1
        clens = torch.cumsum(tmpMatLen, 0)
        idxs = torch.arange(0, torch.sum(tmpMatLen), device=clens.device)
        group = torch.searchsorted(clens, idxs, right=True)
        inc = clens[group]-idxs-1
        ridx = batch_slice[:, 0][group]
        cidx = batch_slice[:, 2][group]-inc
        return group, ridx, cidx
    group, ridx, cidx = getGroupMask(batch_slices)
    arr[ridx, cidx] = val[group]


class CRFLoss(nn.Module):
    def __init__(self, tagset_size):
        super(CRFLoss, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))

    def _logsumexp(self, vec):
        max_score, _ = vec.max(1)
        return max_score + torch.log(torch.exp(vec-max_score.unsqueeze(1)).sum(1))

    def _forward_alg(self, feats, feats_mask):
        _zeros = torch.zeros_like(feats[0]).to(feats.device)
        state = torch.where(feats_mask[0].view(-1, 1), feats[0], _zeros)
        transition_params = self.transitions.unsqueeze(0)
        for i in range(1, feats.shape[0]):
            transition_scores = state.unsqueeze(2) + transition_params
            new_state = feats[i] + self._logsumexp(transition_scores)
            state = torch.where(feats_mask[i].view(-1, 1), new_state, state)
        all_mask = feats_mask.any(0).float()
        return self._logsumexp(state)*all_mask

    def _score_sentence(self, feats, tags, feats_mask):
        # Gives the score of a provided tag sequence
        feats_mask = feats_mask.float()
        time_step, batch_size, tags_size = feats.shape
        s_score = feats.view(-1, tags_size).gather(1, tags.view(-1, 1))*feats_mask.view(-1, 1)
        u_score = s_score.view(-1, batch_size).sum(0)
        if time_step > 1:
            t_mask = feats_mask[:-1].view(-1, 1)*feats_mask[1:].view(-1, 1)
            t_scores = self.transitions.index_select(0, tags[0:-1].view(-1))
            t_score = t_scores.gather(1, tags[1:].view(-1, 1))*t_mask
            u_score += t_score.view(-1, batch_size).sum(0)
        return u_score

    def forward(self, feats, tags, feats_len):
        """feats is [batch,time,tag_size] float tensor
        tags is a int tensor that shape (batch,time)
        feats_len is a int tensor that shape (batch,)
        """
        feats = feats.transpose(0, 1).contiguous()
        tags = tags.long().transpose(0, 1).contiguous()
        base_index = torch.arange(0, feats.shape[0]).unsqueeze(0).expand(feats.shape[1], -1).to(feats.device)
        feats_mask = base_index < feats_len.long().view(-1, 1)
        feats_mask = feats_mask.transpose(0, 1).contiguous()

        forward_score = self._forward_alg(feats, feats_mask)
        gold_score = self._score_sentence(feats, tags, feats_mask)

        return forward_score - gold_score


class LabelSmoothLoss(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothLoss, self).__init__()
        assert 0 <= smoothing < 0.5, "smoothing value must be a in [0,0.5)"
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """x is float shape [N,C]
        target is int shape (N,) and 0<=target[i]<C
        return float tensor shape  (N,)
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.sum(dim=-1)
        beta = self.smoothing/(x.size(-1)-1)
        return (self.confidence-beta) * nll_loss + beta * smooth_loss


class GraphLoss(nn.Module):
    def __init__(self):
        super(GraphLoss, self).__init__()

    def _forward_alg(self, graph, weight):
        # caluate th paths
        paths = {}
        for i, (s, e) in enumerate(graph):
            pth = paths.get(e, [])
            pth.append((s, i))
            paths[e] = pth
        # init data
        esum = torch.zeros(len(paths)+1).to(weight)
        flag = [False]*(len(paths)+1)
        flag[0] = True
        stack = [len(paths)]

        # calculate the prob
        while stack:
            if flag[stack[-1]]:
                # this node has been calculated
                stack.pop()
                continue
            # check sub path
            allpre_ready = True
            for idx, _ in paths[stack[-1]]:
                if not flag[idx]:
                    stack.append(idx)
                    allpre_ready = False
                    continue

            if allpre_ready:
                idx = stack.pop()
                pth = torch.LongTensor(paths[idx]).to(weight.device())
                pre_esums = esum.index_select(0, pth[:, 0])-weight.index_select(0, pth[:, 1])
                esum[idx] = torch.logsumexp(pre_esums, 0)
                flag[idx] = True

        return esum[-1]

    def forward(self, graph, weight):
        """
        graph must has one start idx=0 and one end point=last_idx
        graph is [path_nums,3] int numpy tensor ,every graph path (start,end,bool)
        weight is [path_nums] float torch tensor,every graph path weight
        """
        best_weight_mask = torch.from_numpy(graph[:, 2]).to(weight)
        gold_score = (weight*best_weight_mask).sum(0)

        graph_path = graph[:, :2]
        forward_score = self._forward_alg(graph_path-graph_path.min(), weight)

        return gold_score+forward_score


class SampledSoftMaxCrossEntropy(nn.Module):
    def __init__(self, emb_size, tags_weight, nsampled):
        """
        emb_size is the tag embeding size is int;
        tags_weight is a list that tags
        nsampled how many num of negtive sample
        """
        super(SampledSoftMaxCrossEntropy, self).__init__()
        self.nsampled = nsampled
        self.weight = nn.Parameter(torch.Tensor(len(tags_weight), emb_size))
        self.bias = nn.Parameter(torch.Tensor(len(tags_weight)))
        self.sample_map = self.__genmap__(tags_weight)
        self.loss_fn = nn.CrossEntropyLoss()

    def __genmap__(self, weight):
        freq = np.sum(weight).astype(np.float32)
        weight = np.around(np.power(np.divide(weight, freq), 0.75) * freq).astype(np.int32)
        end_tag = np.cumsum(weight)
        start_tag = np.concatenate([[0], end_tag[:-1]])
        sample_map = np.zeros(end_tag[-1], dtype=np.int32)
        for i, (s, t) in enumerate(start_tag, end_tag):
            sample_map[s:t] = i
        return sample_map

    def forward(self, inputs, labels):
        batchs = labels.shape[0]
        sample_ids = []
        tmpids = set([])
        for lab in labels.cpu().numpy().astype(np.int32):
            while len(tmpids) < self.nsampled:
                tmpids.update(self.sample_map[np.random.randint(self.sample_map.shape[0], size=self.nsampled - len(tmpids))])
                if lab in tmpids:
                    tmpids.remove(lab)
            sample_ids.append(lab)
            sample_ids.extend(tmpids)
            tmpids.clear()

        sample_weights = self.weight[sample_ids, :].reshape((batchs, self.nsampled + 1, -1))
        sample_bias = self.bias[sample_ids].reshape((batchs, self.nsampled + 1))
        logits = torch.einsum("ik,ijk->ij", inputs, sample_weights) + sample_bias
        new_targets = torch.zeros(batchs).long().to(labels.device)
        return self.loss_fn(logits, new_targets)
