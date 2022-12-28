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
import torch
import torch.nn as nn


def getFeatsIdx(postions):
    """postions shape is (N,)
    """
    feats_lens = torch.unique(postions, return_counts=True)[1]
    words_len = torch.max(feats_lens)
    base_idx = torch.arange(0, words_len, device=feats_lens.device).view(1, -1).repeat(feats_lens.shape[0], 1)
    start_idx = torch.roll(torch.cumsum(feats_lens, -1), 1)
    start_idx[0] = 0
    feats_idx = (base_idx + start_idx.view((-1, 1))) * (base_idx < feats_lens.view(-1, 1))
    return feats_lens, feats_idx


def batch_segment_fill(arr, batch_slices, val):
    """arr is a 2d or 3d tensor and shape is (N,T),
    batch_slices is a long tensor and shape is (M,3),
    the meaning of batch_slices's colum is [batch idx,start idx,end idx]
    val is a tensor that shape is (M,)
    """

    def _getGroupMask(batch_slice):
        outlens = batch_slice[:, 2] - batch_slice[:, 1] + 1
        alen, clens = torch.sum(outlens), torch.cumsum(outlens, 0)
        group = torch.zeros(alen, dtype=torch.long, device=outlens.device)
        group[clens[:-1]] = 1
        group = torch.cumsum(group, 0)
        incre_idx = clens[group] - torch.arange(0, alen, device=batch_slice.device, dtype=torch.long) - 1
        ridx = batch_slice[:, 0][group]
        cidx = batch_slice[:, 2][group] - incre_idx
        return group, ridx, cidx

    group, ridx, cidx = _getGroupMask(batch_slices)
    arr[ridx, cidx] = val[group]


class CRFLoss(nn.Module):

    def __init__(self, tagset_size):
        super(CRFLoss, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))

    def viterbi_decode(self, score):
        #score: A [seq_len, num_tags] matrix of unary potentials.
        trellis = torch.zeros_like(score, device=score.device)
        backpointers = torch.zeros_like(score, dtype=torch.long, device=score.device)
        #set data
        seq_length, _ = score.shape
        trellis[0] = score[0]
        for t in range(1, seq_length):
            v = torch.unsqueeze(trellis[t - 1], 1) + self.transitions.data
            trellis[t] = score[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.argmax(v, 0)

        viterbi = torch.zeros((score.shape[0], ), dtype=torch.long, device=score.device)
        viterbi[seq_length - 1] = torch.argmax(trellis[seq_length - 1])
        for t in range(seq_length - 1, 0, -1):
            viterbi[t - 1] = backpointers[t][viterbi[t]]

        return viterbi

    def _logsumexp(self, vec):
        max_score, _ = vec.max(1)
        return max_score + torch.log(torch.exp(vec - max_score.unsqueeze(1)).sum(1))

    def _forward_alg(self, feats, feats_mask):
        _zeros = torch.zeros_like(feats[0]).to(feats.device)
        state = torch.where(feats_mask[0].view(-1, 1), feats[0], _zeros)
        transition_params = self.transitions.unsqueeze(0)
        for i in range(1, feats.shape[0]):
            transition_scores = state.unsqueeze(2) + transition_params
            new_state = feats[i] + self._logsumexp(transition_scores)
            state = torch.where(feats_mask[i].view(-1, 1), new_state, state)
        all_mask = feats_mask.any(0).float()
        return self._logsumexp(state) * all_mask

    def _score_sentence(self, feats, tags, feats_mask):
        # Gives the score of a provided tag sequence
        feats_mask = feats_mask.float()
        time_step, batch_size, tags_size = feats.shape
        s_score = feats.view(-1, tags_size).gather(1, tags.view(-1, 1)) * feats_mask.view(-1, 1)
        u_score = s_score.view(-1, batch_size).sum(0)
        if time_step > 1:
            t_mask = feats_mask[:-1].view(-1, 1) * feats_mask[1:].view(-1, 1)
            t_scores = self.transitions.index_select(0, tags[0:-1].view(-1))
            t_score = t_scores.gather(1, tags[1:].view(-1, 1)) * t_mask
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


class GraphLoss(nn.Module):

    def __init__(self):
        super(GraphLoss, self).__init__()

    @staticmethod
    def _get_dep_order(graph):
        with torch.no_grad():
            nodex = graph.max() + 1
            connc = torch.zeros((nodex, nodex), dtype=torch.float, device=graph.device)
            connc[graph[:, 0], graph[:, 1]] = 1
            initc = torch.zeros(nodex, dtype=torch.long, device=graph.device)
            initc[nodex - 1] = 1
            deg = connc.sum(1)
            while initc.min() < 1:
                trans = torch.matmul((initc > 0).float().view(1, -1), connc.T)
                initc += (trans == deg).long().view(-1)
            return torch.argsort(initc), nodex

    @staticmethod
    def _get_paths(graph):
        with torch.no_grad():
            paths = {}
            for i, (s, e) in enumerate(graph):
                e = int(e.data.cpu())
                pth = paths.get(e, [])
                pth.append((s, i))
                paths[e] = pth
            paths = {k: torch.LongTensor(v).to(graph.device) for k, v in paths.items()}
            return paths

    def _forward_alg(self, graph, weight):
        dep_orders, node_nums = self._get_dep_order(graph)
        paths = self._get_paths(graph)
        # init data
        esum = torch.zeros(node_nums, dtype=weight.dtype, device=weight.device)
        index = torch.arange(node_nums, dtype=torch.long, device=weight.device)
        # calculate the prob
        for idx in dep_orders[1:]:
            dep_points = paths[int(idx.data.cpu())]
            idx_tensor = (index == idx).to(weight.dtype)
            pre_esums = esum.index_select(0, dep_points[:, 0]) - weight.index_select(0, dep_points[:, 1])
            esum += idx_tensor * torch.logsumexp(pre_esums, 0)

        return esum[-1]

    def forward(self, graph, weight):
        """
        graph must has one start idx=0 and one end point=last_idx
        graph is [path_nums,3] int numpy tensor ,every graph path (start,end,bool)
        weight is [path_nums] float torch tensor,every graph path weight
        """
        gold_score = (weight * graph[:, 2].to(weight)).sum(0)
        graph = graph[:, :2] - graph[:, :2].min()
        forward_score = self._forward_alg(graph, weight)
        return gold_score + forward_score
