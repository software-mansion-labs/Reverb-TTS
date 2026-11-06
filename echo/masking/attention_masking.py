import torch


# Masking utility - general bidirectional attention mask
def build_dense_bidirectional_mask(valid: torch.Tensor) -> torch.Tensor:
    """
    [B, L] validity -> [B, 1, L, L] mask, with padding rows kept self-attending.

    A 'validity' is a mask with real tokens marked with boolean True, and padding tokens marked with boolean False.
    """

    batch_size, seq_len = valid.shape
    mask = valid[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).clone()

    # Allow padding positions to attent to itself (preventing NaNs from appearing in softmax)
    diagonal = torch.arange(seq_len, device=valid.device)   # An indexing for the mask slice
    mask[:, :, diagonal, diagonal] = True   # Each position is allowed to attent to itself (even if it is a padding).

    return mask


# Masking utility - prefix blocking fixup
def apply_prefix_blocking_4d(
    attention_mask: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Cuts prefix(query) -> target(key) edges.
    """

    blocked = (~target_mask)[:, None, :, None] & target_mask[:, None, None, :]

    return attention_mask & ~blocked
