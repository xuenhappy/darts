"""Sparse graph objectives used by neural segmentation training."""

import torch
import torch.nn as nn


@torch.no_grad()
def stable_clip_grad_norm_(parameters, max_norm):
    """Clip finite gradients without overflowing the norm reduction in FP32."""
    gradients = [
        parameter.grad for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    non_finite = [gradient for gradient in gradients if not torch.isfinite(gradient).all()]
    if non_finite:
        raise RuntimeError("cannot clip non-finite gradients")
    max_abs = max(float(gradient.detach().abs().max()) for gradient in gradients)
    if max_abs == 0.0:
        return 0.0
    # Scaling before squaring avoids FP32 overflow; FP64 preserves the sum for
    # large candidate graphs with many individually finite gradient elements.
    scaled_square_sum = sum(
        float(((gradient.detach().double() / max_abs) ** 2).sum())
        for gradient in gradients
    )
    total_norm = max_abs * scaled_square_sum ** 0.5
    clip_coefficient = min(1.0, float(max_norm) / (total_norm + 1e-12))
    if clip_coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)
    return total_norm


class _GraphPathNll(torch.autograd.Function):
    """Exact DAG path NLL with an analytic, bounded edge gradient."""

    @staticmethod
    def _logsumexp(values, chunk_size):
        if chunk_size <= 0 or values.numel() <= chunk_size:
            return torch.logsumexp(values, dim=0)
        result = None
        for chunk in values.split(chunk_size):
            chunk_value = torch.logsumexp(chunk, dim=0)
            result = chunk_value if result is None else torch.logaddexp(result, chunk_value)
        return result

    @staticmethod
    def forward(ctx, word_info, graph, association_nll, edge_chunk_size, strict_path):
        output_device = association_nll.device
        # The DP is a Python topological loop and its backward is analytic.
        # Running thousands of scalar logsumexp/exp kernels on CUDA is slower,
        # retains allocator workspaces, and has destabilized consumer drivers.
        # Compute detached path posteriors on CPU and return only the bounded
        # edge gradient to the accelerator.
        work_nll = association_nll.detach().cpu() if association_nll.is_cuda else association_nll
        work_info = word_info.detach().cpu() if word_info.is_cuda else word_info
        work_graph = graph.detach().cpu() if graph.is_cuda else graph
        dtype = work_nll.dtype
        batch_ids = work_info[:, 0].long()
        batch_num = int(batch_ids.max().item()) + 1
        counts = torch.bincount(batch_ids, minlength=batch_num)
        offsets = torch.zeros_like(counts)
        if counts.numel() > 1:
            offsets[1:] = torch.cumsum(counts[:-1], 0)

        total_loss = work_nll.new_zeros(())
        edge_gradient = torch.zeros_like(work_nll)
        valid_graphs = 0
        for batch_id in range(batch_num):
            node_num = int(counts[batch_id].item())
            if node_num == 0:
                continue
            node_offset = int(offsets[batch_id].item())
            edge_indexes = torch.nonzero(
                work_graph[:, 0].long() == batch_id, as_tuple=False
            ).flatten()
            if edge_indexes.numel() == 0:
                if strict_path and node_num > 1:
                    raise RuntimeError(f"graph {batch_id} has nodes but no edges")
                continue

            edges = work_graph[edge_indexes]
            edge_nll = work_nll[edge_indexes]
            sources = (edges[:, 1].long() - node_offset).tolist()
            destinations = (edges[:, 2].long() - node_offset).tolist()
            incoming = [[] for _ in range(node_num)]
            outgoing = [[] for _ in range(node_num)]
            for edge_id, (source, destination) in enumerate(zip(sources, destinations)):
                if 0 <= source < destination < node_num:
                    incoming[destination].append(edge_id)
                    outgoing[source].append(edge_id)

            neg_inf = work_nll.new_tensor(float("-inf"))
            alpha = [neg_inf for _ in range(node_num)]
            alpha[0] = work_nll.new_zeros(())
            forward_reachable = [False for _ in range(node_num)]
            forward_reachable[0] = True
            for destination in range(1, node_num):
                valid = [
                    edge_id for edge_id in incoming[destination]
                    if forward_reachable[sources[edge_id]]
                ]
                if not valid:
                    continue
                values = torch.stack([
                    alpha[sources[edge_id]] - edge_nll[edge_id]
                    for edge_id in valid
                ])
                alpha[destination] = _GraphPathNll._logsumexp(
                    values, edge_chunk_size
                )
                forward_reachable[destination] = True

            if not forward_reachable[-1]:
                if strict_path:
                    raise RuntimeError(f"no valid path found for graph {batch_id}")
                continue
            log_partition = alpha[-1]
            if not torch.isfinite(log_partition):
                raise RuntimeError(f"non-finite path cost found for graph {batch_id}")

            beta = [neg_inf for _ in range(node_num)]
            beta[-1] = work_nll.new_zeros(())
            backward_reachable = [False for _ in range(node_num)]
            backward_reachable[-1] = True
            for source in range(node_num - 2, -1, -1):
                valid = [
                    edge_id for edge_id in outgoing[source]
                    if backward_reachable[destinations[edge_id]]
                ]
                if not valid:
                    continue
                values = torch.stack([
                    -edge_nll[edge_id] + beta[destinations[edge_id]]
                    for edge_id in valid
                ])
                beta[source] = _GraphPathNll._logsumexp(
                    values, edge_chunk_size
                )
                backward_reachable[source] = True

            gold_mask = edges[:, 3].to(dtype)
            local_gradient = gold_mask.clone()
            for edge_id, (source, destination) in enumerate(zip(sources, destinations)):
                if (0 <= source < destination < node_num
                        and forward_reachable[source]
                        and backward_reachable[destination]):
                    log_posterior = (
                        alpha[source] - edge_nll[edge_id]
                        + beta[destination] - log_partition
                    )
                    local_gradient[edge_id] -= torch.exp(log_posterior)

            total_loss += (edge_nll * gold_mask).sum() + log_partition
            edge_gradient[edge_indexes] += local_gradient
            valid_graphs += 1

        if valid_graphs:
            total_loss /= valid_graphs
            edge_gradient /= valid_graphs
        edge_gradient = edge_gradient.to(output_device)
        ctx.save_for_backward(edge_gradient)
        return total_loss.to(output_device)

    @staticmethod
    def backward(ctx, grad_output):
        (edge_gradient,) = ctx.saved_tensors
        return None, None, grad_output * edge_gradient, None, None


class GraphLossSparse(nn.Module):
    """Conditional path NLL with O(E) memory for a batch of candidate DAGs.

    Every edge weight is ``-log P(dst is associated with src)``.  For gold path
    G the objective is ``cost(G) + log(sum_path(exp(-cost(path))))``.  Therefore
    lower quantizer output always means a more likely transition, not an
    arbitrary score.

    Backward does not differentiate through the sequential dynamic program.
    Exact forward/backward path posteriors give each edge the analytic gradient
    ``gold_mask - posterior``, preserving all non-gold supervision while
    bounding the edge gradient to ``[-1, 1]``.

    ``word_info`` rows start with a batch id. ``graph`` rows are
    ``[batch_id, global_src, global_dst, gold_mask]`` and should be sorted by
    ``(batch_id, global_dst, global_src)``.  Nodes must be topologically ordered
    inside each graph, with head first and tail last.
    """

    def __init__(self, edge_chunk_size=65536, strict_path=True):
        super().__init__()
        self.edge_chunk_size = int(edge_chunk_size) if edge_chunk_size else 0
        self.strict_path = strict_path

    def forward(self, word_info, graph, association_nll):
        if association_nll is None:
            return torch.zeros(())
        if (word_info is None or graph is None or word_info.numel() == 0 or
                graph.numel() == 0 or association_nll.numel() == 0):
            return torch.zeros((), device=association_nll.device, dtype=association_nll.dtype)

        return _GraphPathNll.apply(
            word_info, graph, association_nll,
            self.edge_chunk_size, self.strict_path,
        )
