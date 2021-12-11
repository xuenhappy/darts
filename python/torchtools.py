
import torch.nn as nn
import torch


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
    #nn.utils.clip_grad_norm_(model.parameters(), 1, 'inf')
    # for name, p in model.named_parameters():
    #     print(name, p.grad)
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
    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)
    res, _ = rnn(packed_inputs)
    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True, total_length=inputs.shape[1])
    return padded_res.index_select(0, desorted_indices).contiguous()


def segment_max_pool(embeding, idx):
    """
    embededing is a float tensor and shape is (N,T,C),
    idx is a long tensor and shape is (M,3) ,the colum of this tensor is [batch idx,start idx,end idx]
    """
    mask = torch.arange(0, embeding.size(1)).repeat(idx.size(0), 1)
    mask = mask.to(idx.device)
    mask = (mask < idx[:, 1].view(-1, 1)) | (mask > idx[:, 2].view(-1, 1))
    v_mask = torch.zeros_like(mask, dtype=torch.float32).to(mask.device)
    v_mask[mask] = float('-inf')
    v_mask = v_mask.to(embeding.device).view(-1, 1)
    base = embeding.index_select(0, idx[:, 0])
    mask_value = (base.view(-1, base.size(2))+v_mask).view(base.shape)
    return mask_value.max(1)[0]
