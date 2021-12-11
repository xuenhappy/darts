import torch
import torch.nn as nn
import numpy as np


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


class CRFLoss(nn.Module):

    def __init__(self, tagset_size):
        super(CRFLoss, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))

    def _forward_alg(self, feats, feats_mask):
        _zeros = torch.zeros_like(feats[0]).to(feats.device)
        state = torch.where(feats_mask[0].view(-1, 1), feats[0], _zeros)
        transition_params = self.transitions.unsqueeze(0)
        for i in range(1, feats.shape[0]):
            transition_scores = state.unsqueeze(2) + transition_params
            new_state = feats[i] + torch.logsumexp(transition_scores, 1)
            state = torch.where(feats_mask[i].view(-1, 1), new_state, state)
        all_mask = feats_mask.any(0).float()
        return torch.logsumexp(state, 1) * all_mask

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
        # feats is [batch,time,tag_size] float tensor
        # tags is [batch,time]  tensor
        # feats_len is [batch,]  tensor
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

    def _forward_alg(self, graph, weight):
        # caluate th paths
        paths = {}
        for i, (s, e, _) in enumerate(graph):
            pth = paths.get(e, [])
            pth.append((s, i))
            paths[e] = pth
        # init data
        esum = torch.zeros(len(paths)+1).float().to(weight.device())
        flag = [False]*(len(paths)+1)
        flag[0] = True
        stack = [len(paths)]

        # calculate the prob
        while stack:
            if flag[stack[-1]]:
                # this node hase been calculated
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
        graph = graph.astype(int)
        gold_score = (weight*torch.from_numpy(graph[:, 2]).to(weight.device).float()).sum(0)
        forward_score = self._forward_alg(graph, weight)
        return gold_score+forward_score
