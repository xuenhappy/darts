"""Sparse graph objectives used by neural segmentation training."""

import torch
import torch.nn as nn


class GraphLossSparse(nn.Module):
    """Conditional path NLL with O(E) memory for a batch of candidate DAGs.

    Every edge weight is ``-log P(dst is associated with src)``.  For gold path
    G the objective is ``cost(G) + log(sum_path(exp(-cost(path))))``.  Therefore
    lower quantizer output always means a more likely transition, not an
    arbitrary score.

    ``word_info`` rows start with a batch id. ``graph`` rows are
    ``[batch_id, global_src, global_dst, gold_mask]`` and should be sorted by
    ``(batch_id, global_dst, global_src)``.  Nodes must be topologically ordered
    inside each graph, with head first and tail last.
    """

    def __init__(self, edge_chunk_size=65536, strict_path=True):
        super().__init__()
        self.edge_chunk_size = int(edge_chunk_size) if edge_chunk_size else 0
        self.strict_path = strict_path

    @staticmethod
    def _chunked_logsumexp(values, chunk_size):
        if values.numel() == 0:
            return values.new_tensor(float("-inf"))
        if chunk_size <= 0 or values.numel() <= chunk_size:
            return torch.logsumexp(values, dim=0)
        result = None
        for chunk in values.split(chunk_size):
            chunk_value = torch.logsumexp(chunk, dim=0)
            result = chunk_value if result is None else torch.logaddexp(result, chunk_value)
        return result

    def forward(self, word_info, graph, association_nll):
        if association_nll is None:
            return torch.zeros(())
        if (word_info is None or graph is None or word_info.numel() == 0 or
                graph.numel() == 0 or association_nll.numel() == 0):
            return torch.zeros((), device=association_nll.device, dtype=association_nll.dtype)

        device = association_nll.device
        dtype = association_nll.dtype
        batch_ids = word_info[:, 0].long()
        batch_num = int(batch_ids.max().item()) + 1
        counts = torch.bincount(batch_ids, minlength=batch_num)
        offsets = torch.zeros_like(counts)
        if counts.numel() > 1:
            offsets[1:] = torch.cumsum(counts[:-1], 0)

        total_loss = torch.zeros((), device=device, dtype=dtype)
        valid_graphs = 0
        for batch_id in range(batch_num):
            node_num = int(counts[batch_id].item())
            if node_num == 0:
                continue
            node_offset = int(offsets[batch_id].item())
            edge_mask = graph[:, 0].long() == batch_id
            if not torch.any(edge_mask):
                if self.strict_path and node_num > 1:
                    raise RuntimeError(f"graph {batch_id} has nodes but no edges")
                continue

            edges = graph[edge_mask]
            edge_nll = association_nll[edge_mask]
            gold_mask = edges[:, 3].to(dtype)
            local_src = edges[:, 1].long() - node_offset
            local_dst = edges[:, 2].long() - node_offset
            if torch.any(local_dst[1:] < local_dst[:-1]):
                order = torch.argsort(local_dst * node_num + local_src)
                local_src = local_src[order]
                local_dst = local_dst[order]
                edge_nll = edge_nll[order]
                gold_mask = gold_mask[order]

            # A tensor list avoids in-place updates of an autograd-tracked DP
            # array. Each value is the log partition of paths ending at a node.
            sources = local_src.tolist()
            neg_inf = association_nll.new_tensor(float("-inf"))
            partition = [neg_inf for _ in range(node_num)]
            partition[0] = association_nll.new_zeros(())
            position = 0
            edge_num = local_dst.numel()
            while position < edge_num:
                destination = int(local_dst[position].item())
                end = position + 1
                while end < edge_num and int(local_dst[end].item()) == destination:
                    end += 1
                if 0 <= destination < node_num:
                    previous = torch.stack([partition[src] for src in sources[position:end]])
                    values = previous - edge_nll[position:end]
                    partition[destination] = self._chunked_logsumexp(values, self.edge_chunk_size)
                position = end

            log_partition = partition[-1]
            if not torch.isfinite(log_partition):
                if self.strict_path:
                    raise RuntimeError(f"no valid path found for graph {batch_id}")
                continue
            gold_cost = (edge_nll * gold_mask).sum()
            total_loss = total_loss + gold_cost + log_partition
            valid_graphs += 1

        if valid_graphs == 0:
            return torch.zeros((), device=device, dtype=dtype)
        return total_loss / valid_graphs
