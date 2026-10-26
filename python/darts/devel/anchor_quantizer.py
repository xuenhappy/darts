"""Darts gauge-anchored decomposed energy quantizer.

All edge costs are nonnegative and preserve the existing GraphLossSparse input
contract.  Destination unary potential is counted once per visited node.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AnchorDecomposedQuantizer(nn.Module):
    """c(u,v) = softplus(-a_unary(v)) + softplus(-a_pair(u,v)).

    The Q/K path is asymmetric; the unary head only sees the destination.
    Head/tail remain in the existing word encoding convention. Tail unary is
    path-independent within a graph, so it cannot change path preferences.

    Export remains compatible with current two-input Quantizer.export2onnx.
    """
    def __init__(self, input_size, hidden_size=32, distance_buckets=0,
                 unary_init_cost=0.7, pair_init_cost=1.3):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError('input_size and hidden_size must be positive')
        if unary_init_cost <= 0 or pair_init_cost <= 0:
            raise ValueError('initial costs must be positive')
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.distance_buckets = int(distance_buckets)
        self.score = 'anchored-decomposed'
        self.max_nll = None  # no artificial 50 ceiling
        self.Kmap = nn.Linear(input_size, hidden_size, bias=False)
        self.Qmap = nn.Linear(input_size, hidden_size, bias=False)
        self.unary_head = nn.Linear(input_size, 1)
        self.pair_bias = nn.Parameter(torch.zeros(()))
        self.log_pair_scale = nn.Parameter(torch.zeros(()))
        if distance_buckets > 0:
            self.distance_embedding = nn.Embedding(distance_buckets, input_size)
            nn.init.zeros_(self.distance_embedding.weight)
        nn.init.zeros_(self.unary_head.weight)
        nn.init.constant_(self.unary_head.bias, -self._inverse_softplus(unary_init_cost))
        # The zero-initialized bias anchors pairwise energy near its target at
        # init. q/k embeddings remain trainable, unlike setting both to zero.
        with torch.no_grad():
            self.pair_bias.fill_(-self._inverse_softplus(pair_init_cost))

    @staticmethod
    def _inverse_softplus(y):
        return math.log(math.expm1(y))

    def unary_cost(self, dst):
        return F.softplus(-self.unary_head(dst).squeeze(-1))

    def pair_logit(self, src, dst, gaps=None):
        if gaps is not None and self.distance_buckets > 0:
            d = self.distance_embedding(gaps)
            src, dst = src + d, dst + d
        dot = (self.Kmap(src) * self.Qmap(dst)).sum(dim=-1)
        # A bounded trainable scaling avoids unconstrained growth of dot logit.
        scale = torch.exp(self.log_pair_scale.clamp(-3.0, 3.0)) / math.sqrt(self.hidden_size)
        return dot * scale + self.pair_bias

    def pair_cost(self, src, dst, gaps=None):
        return F.softplus(-self.pair_logit(src, dst, gaps))

    def components(self, src, dst, gaps=None):
        return self.unary_cost(dst), self.pair_cost(src, dst, gaps)

    def forward(self, src, dst, gaps=None):
        u, t = self.components(src, dst, gaps)
        return (u + t).reshape(-1)

    def logits(self, src, dst, gaps=None):
        """Exact local Bernoulli logit associated with P(edge)=exp(-cost).

        For c>0: logit(exp(-c)) = -c - log(1-exp(-c)).
        torch.expm1 avoids catastrophic cancellation for small c.
        Note: local BCE is auxiliary; the path distribution stays globally
        normalized by GraphLossSparse.
        """
        c = self.forward(src, dst, gaps).float().clamp_min(1e-7)
        return -c - torch.log(-torch.expm1(-c))

    def export2onnx(self, outfile='quantizer.onnx'):
        """Two-input export when distance_buckets == 0 (current MNN ABI)."""
        if self.distance_buckets:
            raise NotImplementedError('distance embedding requires a 3-input MNN runtime ABI')
        class Exporter(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, a, b):
                return self.model(a, b)
        dummy = (torch.randn(3, self.input_size), torch.randn(3, self.input_size))
        torch.onnx.export(
            Exporter(self.eval()), dummy, outfile,
            input_names=['a', 'b'], output_names=['association_nll'],
            opset_version=18,
            dynamic_axes={'a': {0: 'edges'}, 'b': {0: 'edges'},
                          'association_nll': {0: 'edges'}},
        )
        return outfile


def atom_anchor_and_ratio_loss(edge_costs, graph, *, atom_target=2.0,
                               ratio_weight=0.05, anchor_weight=0.02,
                               anchor_all_atom_gold=False):
    """Optional additions to exact DAG GraphLoss.

    graph[:,0] is batch id, [:,3] gold edge mask, [:,4] atom edge mask.
    Atom anchor: (mean_atom_cost - atom_target)^2, invariant to sentence
    length after per-graph normalization. Ratio: existing Darts expression
    (E_gold/sqrt(N_gold))/(E_atom/sqrt(N_atom)), per graph.

    Anchor and ratio are disabled on graphs where gold_path == atom_path by
    default: imposing a high cost anchor on a true all-atom segmentation is
    contradictory to its role as an intentionally expensive competitor.
    """
    if graph.ndim != 2 or graph.shape[1] < 5:
        raise ValueError('graph needs >=5 columns: batch, src, dst, gold, atom')
    if edge_costs.ndim != 1 or edge_costs.numel() != graph.shape[0]:
        raise ValueError('edge_costs must be [edge_count]')
    if atom_target <= 0 or ratio_weight < 0 or anchor_weight < 0:
        raise ValueError('atom_target must be positive and weights nonnegative')
    zero = edge_costs.sum() * 0.0
    if graph.numel() == 0:
        return zero, {'ratio': zero.detach(), 'anchor': zero.detach(), 'active': 0}

    gids = graph[:, 0].long()
    n_graphs = int(gids.max().item()) + 1
    gold = graph[:, 3].bool()
    atom = graph[:, 4].bool()

    def reduce_mask(mask):
        counts = edge_costs.new_zeros(n_graphs)
        sums = edge_costs.new_zeros(n_graphs)
        f = mask.to(edge_costs.dtype)
        counts.index_add_(0, gids, f)
        sums.index_add_(0, gids, edge_costs * f)
        return sums, counts

    gold_sum, gold_count = reduce_mask(gold)
    atom_sum, atom_count = reduce_mask(atom)
    diff = edge_costs.new_zeros(n_graphs)
    diff.index_add_(0, gids, (gold != atom).to(edge_costs.dtype))
    have_paths = (gold_count > 0) & (atom_count > 0)
    distinct = have_paths & (diff > 0)
    ratio_valid = distinct & (gold_count > 1.5) & (atom_count > 1.5) & (atom_sum > 1e-3)
    anchor_valid = have_paths if anchor_all_atom_gold else distinct
    if bool(ratio_valid.any()):
        A = gold_sum[ratio_valid] / gold_count[ratio_valid].sqrt()
        B = atom_sum[ratio_valid] / atom_count[ratio_valid].sqrt()
        ratio = (A / B).mean()
    else:
        ratio = zero
    if bool(anchor_valid.any()):
        atom_mean = atom_sum[anchor_valid] / atom_count[anchor_valid]
        anchor = (atom_mean - atom_target).square().mean()
    else:
        anchor = zero
    return ratio_weight * ratio + anchor_weight * anchor, {
        'ratio': ratio.detach(), 'anchor': anchor.detach(),
        'active': int(anchor_valid.sum().item())}
