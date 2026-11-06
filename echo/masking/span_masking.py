import torch

from typing import Tuple


# Helper function - span lengths to positions expand
def _span_bounds(
    prefix_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-row prefix end and target end, plus the batch's position row.
    """

    target_starts = prefix_lengths.to(device).unsqueeze(1)
    target_ends = target_starts + target_lengths.to(device).unsqueeze(1)
    positions = torch.arange(int(target_ends.max()), device=device).unsqueeze(0)
    
    return positions, target_starts, target_ends


# Masking utility - general prefix mask
def build_prefix_mask(
    prefix_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    [B] spans -> [B, L] mask of the prefix positions.
    """
    
    positions, starts, _ = _span_bounds(prefix_lengths, target_lengths, device)
    
    return positions < starts


# Masking utility - general target mask
def build_target_mask(
    prefix_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    [B] spans -> [B, L] mask of the target positions.
    """

    positions, starts, ends = _span_bounds(prefix_lengths, target_lengths, device)

    return (positions >= starts) & (positions < ends)


def target_spans(
    target_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Start index and length of each row's target span.
    """

    lengths = target_mask.sum(dim=1)
    starts = torch.argmax(target_mask.int(), dim=1)

    return starts.where(lengths > 0, torch.zeros_like(starts)), lengths


def gather_target(
    x: torch.Tensor,
    prefix_lengths: torch.Tensor,
    target_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slice each row's target span out of ``x`` ([B, C, L] or [B, C, L, V]).
    """

    max_len = int(target_lengths.max())
    offsets = torch.arange(max_len, device=x.device).unsqueeze(0)
    valid = offsets < target_lengths.unsqueeze(1)
    index = (prefix_lengths.unsqueeze(1) + offsets).clamp(max=x.size(2) - 1)

    index = index[:, None, :]
    while index.dim() < x.dim():
        index = index.unsqueeze(-1)

    return x.gather(2, index.expand(*x.shape[:2], max_len, *x.shape[3:])), valid
