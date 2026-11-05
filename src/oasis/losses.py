from __future__ import annotations

import torch
import torch.nn.functional as F


def adjacent_targets(scores: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:

    scores = scores.to(device=anchors.device, dtype=torch.float32)
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Scores must be a finite one-dimensional tensor")
    if anchors.ndim != 1 or anchors.numel() < 2 or not (anchors.diff() > 0).all():
        raise ValueError("Anchors must be strictly increasing")
    if ((scores < anchors[0]) | (scores > anchors[-1])).any():
        raise ValueError("Scores fall outside the score anchors")
    upper = torch.searchsorted(anchors, scores.contiguous()).clamp(1, anchors.numel() - 1)
    lower = upper - 1
    fraction = (scores - anchors[lower]) / (anchors[upper] - anchors[lower])
    target = torch.zeros((scores.numel(), anchors.numel()), device=anchors.device)
    target.scatter_add_(1, lower[:, None], (1 - fraction)[:, None])
    target.scatter_add_(1, upper[:, None], fraction[:, None])
    return target


def score_loss(logits: torch.Tensor, scores: torch.Tensor, token_ids: torch.Tensor,
               anchors: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if logits.ndim != 2 or logits.shape[0] != scores.numel():
        raise ValueError("Expected one full-vocabulary logit vector per score")
    logits = logits.float()
    token_ids = token_ids.to(device=logits.device, dtype=torch.long)
    anchors = anchors.to(device=logits.device, dtype=torch.float32)
    if token_ids.ndim != 1 or token_ids.numel() != anchors.numel():
        raise ValueError("Token IDs and anchors must have the same length")
    if token_ids.unique().numel() != token_ids.numel():
        raise ValueError("Score token IDs must be distinct")
    target = adjacent_targets(scores, anchors)
    log_probabilities = F.log_softmax(logits, dim=-1).index_select(-1, token_ids)
    cross_entropy = -(target * log_probabilities).sum(-1).mean()
    probabilities = logits.index_select(-1, token_ids).softmax(-1)
    distance = (probabilities.cumsum(-1) - target.cumsum(-1)).abs()[:, :-1]
    wasserstein = (distance * anchors.diff()).sum(-1).mean() / 100.0
    return cross_entropy + 0.5 * wasserstein, {
        "soft_ce": cross_entropy.detach(),
        "ordinal_w1": wasserstein.detach(),
    }
