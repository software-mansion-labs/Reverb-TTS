import torch

from typing import Any, Dict, List, Optional

from echo.data.constants import AUDIO_MASK_ID, IGNORE_INDEX
from echo.masking.attention_masking import build_dense_bidirectional_mask
from echo.masking.constants import MAIN_CODEBOOKS
from echo.masking.span_masking import gather_target


def build_model_inputs(
    batch: Dict[str, List[Any]],                   # A collated batch, as produced by echo.data.loader.collate
    masks: Optional[List[torch.Tensor]] = None,    # [C, T] per sample, the positions to mask and score
    polish: Optional[List[bool]] = None,           # Per sample, a polish state - otherwise a main-path one, scored on codebooks 0-1 alone
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Assembles a collated batch into the inputs :meth:`EchoModel.forward` takes.
    """

    rows, label_rows, audio_flags = [], [], []

    for index, audio in enumerate(batch["audio_tokens"]):
        prefix = torch.cat([batch["prompt_tokens"][index], batch["text_tokens"][index]])
        labels = torch.full(audio.shape, IGNORE_INDEX, dtype=torch.long)

        if masks is not None:
            mask = masks[index]
            labels[mask] = audio[mask]
            audio = audio.masked_fill(mask, AUDIO_MASK_ID)

            # The main path never commits the codebooks the polish pass fills, so it is not scored on them.
            if polish is not None and not polish[index]:
                labels[MAIN_CODEBOOKS:] = IGNORE_INDEX

        # The prompt and the text are shared by every codebook row.
        text_block = prefix.unsqueeze(0).expand(audio.size(0), -1)

        rows.append(torch.cat([text_block, audio], dim=1))
        label_rows.append(torch.cat([torch.full_like(text_block, IGNORE_INDEX), labels], dim=1))
        audio_flags.append(
            torch.arange(prefix.numel() + audio.size(1)) >= prefix.numel()
        )

    input_ids = torch.stack(rows)
    positions = torch.arange(input_ids.size(-1)).unsqueeze(0)

    inputs = {
        "input_ids": input_ids,
        "audio_mask": torch.stack(audio_flags),
        "attention_mask": build_dense_bidirectional_mask(batch["valid"]),
        "position_ids": positions.expand(input_ids.size(0), -1),
        "prefix_lengths": torch.tensor(batch["prefix_length"]),
        "target_lengths": torch.tensor(batch["target_length"]),
    }

    if masks is not None:
        inputs["labels"] = torch.stack(label_rows)

    return {key: value.to(device) for key, value in inputs.items()} if device else inputs


def build_unconditional_inputs(
    input_ids: torch.Tensor,   # [B, C, L] the conditional pass's ids
    starts: torch.Tensor,      # [B] where each row's target span begins
    lengths: torch.Tensor,     # [B] how long it is
) -> Dict[str, torch.Tensor]:
    """
    The unconditional pass's inputs: each row's target span alone, with neither text nor prefix.

    Built the way EchoModel._uncond_logits builds them - a test holds the two together.
    """

    target_ids, valid = gather_target(input_ids, starts, lengths)
    positions = torch.arange(target_ids.size(-1), device=input_ids.device)

    return {
        "input_ids": torch.where(valid.unsqueeze(1), target_ids, AUDIO_MASK_ID),
        "audio_mask": valid,
        "attention_mask": build_dense_bidirectional_mask(valid),
        "position_ids": positions.unsqueeze(0).expand(target_ids.size(0), -1),
    }
