import torch
import torch.nn.functional as F

from typing import Optional


def guided_log_probs(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    guidance_scale: float,
    mask_id: Optional[int] = None,   # Token to rule out, the way the sampler rules out the mask
) -> torch.Tensor:
    """
    Applies CFG mixture by mixing conditional and unconditional logits.
    """

    cond = F.log_softmax(cond_logits, dim=-1)
    uncond = F.log_softmax(uncond_logits, dim=-1)
    mixed = torch.log_softmax(cond + guidance_scale * (cond - uncond), dim=-1)

    if mask_id is None:
        return mixed

    # Zero-out the mask token probabilities - as it is not a valid token to be used in output.
    mixed[..., mask_id] = -float("inf")

    return torch.log_softmax(mixed, dim=-1)
