"""Token-normalized causal loss for Trainer's distributed accumulation contract.

Qwen3-VL in Transformers 4.57.6 does not forward num_items_in_batch to its own
loss. Supply this callable as Trainer(compute_loss_func=...) with
average_tokens_across_devices=True. Trainer obtains the denominator across the
entire accumulation window and all ranks, and supplies the DDP world-size factor.
This function must neither all-reduce nor multiply/divide by world size again.
"""
from __future__ import annotations


def supervised_logit_positions(labels):
    """Sequence positions whose next-token predictions are supervised in this batch."""
    import torch
    if labels is None or labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Causal labels require a two-dimensional sequence with context")
    if bool(labels[:, 0].ne(-100).any()):
        raise ValueError("The first sequence label must be masked context")
    positions = torch.nonzero(labels[:, 1:].ne(-100).any(dim=0), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("The batch contains no supervised next-token predictions")
    return positions


def selective_qwen_forward(model, labels):
    """Keep full multimodal attention, but avoid vocabulary logits for ignored history.

    Qwen3-VL 4.57.6 supports a tensor of sequence indices in logits_to_keep.
    Trainer still owns loss normalization and calls its configured loss function.
    The closure invokes the original wrapped model once (including DDP/AMP hooks),
    and labels the compact logits with their exact original prediction positions.
    """
    positions = supervised_logit_positions(labels)

    def forward(**inputs):
        if "logits_to_keep" in inputs:
            raise ValueError("Selective training owns logits_to_keep; do not override it")
        outputs = model(**inputs, logits_to_keep=positions)
        outputs["causal_logit_positions"] = positions
        return outputs

    return forward


def causal_token_loss(outputs, labels, num_items_in_batch=None):
    import torch
    import torch.nn.functional as functional

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    if labels is None or labels.ndim != 2 or logits.ndim != 3 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Causal token loss requires [batch, sequence] labels and matching batched logits")
    if labels.shape[1] < 2:
        raise ValueError("A causal example requires a context and a target token")
    # Admission/collation guarantees a masked context precedes every target.
    # Trainer counts unshifted nonignored labels, so an unmasked first label
    # would otherwise produce a different denominator from the causal objective.
    if bool(labels[:, 0].ne(-100).any()):
        raise ValueError("The first sequence label must be masked context")
    positions = outputs.get("causal_logit_positions") if isinstance(outputs, dict) else getattr(outputs, "causal_logit_positions", None)
    if positions is None:
        if logits.shape[:2] != labels.shape:
            raise ValueError("Full logits must match the label sequence; compact logits require explicit positions")
        shifted_labels = labels[:, 1:].to(logits.device)
        prediction_logits = logits[:, :-1, :]
    else:
        expected = supervised_logit_positions(labels)
        if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.long
                or positions.ndim != 1 or not torch.equal(positions.to(expected.device), expected)
                or logits.shape[1] != positions.numel()):
            raise ValueError("Compact logits must cover exactly the supervised prediction positions in order")
        shifted_labels = labels.index_select(1, expected + 1).to(logits.device)
        prediction_logits = logits
    supervised = shifted_labels.ne(-100)
    # Select supervised positions before fp32 upcasting. Visual and history
    # positions can dominate a long context; their vocabulary logits need not
    # allocate another full fp32 tensor merely to be ignored by cross entropy.
    selected_logits = prediction_logits[supervised].float()
    loss_sum = functional.cross_entropy(selected_logits, shifted_labels[supervised], reduction="sum")
    count = supervised.sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(count, device=loss_sum.device, dtype=loss_sum.dtype)
    if denominator.numel() != 1 or not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
        raise ValueError("Supervised token denominator must be one positive finite value")
    return loss_sum / denominator.reshape(())
