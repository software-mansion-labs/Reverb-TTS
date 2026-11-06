import numpy as np
import torch

from dataclasses import dataclass
from transformers import PreTrainedModel
from typing import Any, List, Optional, Union

from echo.data.constants import AUDIO_MASK_ID
from echo.masking.attention_masking import (
    apply_prefix_blocking_4d,
    build_dense_bidirectional_mask,
)
from echo.masking.span_masking import build_target_mask, gather_target, target_spans
from echo.model.config import EchoConfig
from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
    OmniVoiceModelOutput,
)

# Expands base model's return type with the two CFG branches.
# The mixture itself is left to the caller - see echo/utils/cfg_utils.py.
# NOTE: when prefix is dropped, uncond_logits stays None, but cond_logits are equivalent to them
@dataclass
class EchoModelOutput(OmniVoiceModelOutput):
    cond_logits: Optional[torch.Tensor] = None  # [B, C, T, V] conditional logits, over the target span alone
    uncond_logits: Optional[torch.Tensor] = None  # [B, C, T, V] the same span, seen without its prefix
    target_valid: Optional[torch.Tensor] = None  # [B, T] marks the real positions of that span


# The audio heads, centered per codebook.
class CenteredAudioHeads(torch.nn.Linear):
    """
    The audio heads, with every codebook's mean row taken out of its rows before the product.

    What all logits of one codebook share leaves its softmax unchanged, so training never kept it small - on the
    fine codebooks it reaches ~130, where bf16 holds only whole numbers and rounds the distribution away.
    Centering removes it exactly: the probabilities stay the same, for any checkpoint.
    """

    def __init__(self, heads: torch.nn.Linear, codebooks: int) -> None:
        super().__init__(heads.in_features, heads.out_features, bias=False, device="meta")
        # The same parameter, so its name, its initialization and the checkpoint key stay as they were.
        self.weight = heads.weight
        self.codebooks = codebooks

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight.view(self.codebooks, -1, self.in_features)
        centered = weight - weight.mean(dim=1, keepdim=True)

        return torch.nn.functional.linear(hidden_states, centered.view_as(self.weight))


# The text embedding, factorized.
class FactorizedEmbedding(torch.nn.Module):
    """
    The text embedding as a rank-r product: a [V, r] table, then an r -> H projection, in place of a [V, H] table.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.table = torch.nn.Embedding(num_embeddings, rank)
        self.project = torch.nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.project(self.table(input_ids))


# Main model class.
class EchoModel(OmniVoice):
    config_class = OmniVoiceConfig

    def __init__(
        self,
        config: Union[EchoConfig, OmniVoiceConfig],
        llm: Optional[PreTrainedModel] = None,  # Purely optional, allows to load a different pretrained LLM backbone
    ) -> None:
        if isinstance(config, EchoConfig):
            # Loading from an EchoConfig is the desired behaviuor.
            echo_config = config
            omnivoice_config = config.to_omnivoice_config()
        else:
            # If plain OmniVoice config is passed, all the extra attributes (prefix blocking, cfg, etc)
            # are set to default values.
            echo_config = EchoConfig.from_omnivoice_config(config)
            omnivoice_config = config

        super().__init__(omnivoice_config, llm=llm)

        self.audio_heads = CenteredAudioHeads(self.audio_heads, omnivoice_config.num_audio_codebook)
        rank = echo_config.architecture.embedding_rank
        if rank is not None:
            llm_config = omnivoice_config.llm_config
            self.llm.embed_tokens = FactorizedEmbedding(llm_config.vocab_size, llm_config.hidden_size, rank)
        self.echo_config = echo_config
        self.prefix_blocked = echo_config.prefix_blocking.enabled
        self.unconditional = echo_config.cfg.unconditional_pass
        self.guidance_scale = echo_config.cfg.generation_scale

    # Single Pass
    # Used for training / distilation purposes.
    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, C, L] audio + text token ids - text rows are duplicated, audio rows hold one codebook each
        audio_mask: torch.Tensor,  # [B, L], marks audio token positions, allowing to exclude them from the text
        labels: Optional[torch.LongTensor] = None,  # [B, C, L] optional target token ids, causing the CE loss to be computed. -100 where no loss is computed.
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, L, L] bidirectional mask, passed to the LLM backbone as is
        document_ids: Optional[torch.Tensor] = None,  # [1, L] sample each token belongs to when samples are packed into one row (-1 = padding)
        position_ids: Optional[torch.LongTensor] = None,  # [B, L] position of each token within its own sample
        *,
        target_mask: Optional[torch.Tensor] = None,  # An option to pass an already crafted target mask - useful for training scripts
        prefix_lengths: Optional[torch.LongTensor] = None,  # If target_mask is not given, we need this to calculate it
        target_lengths: Optional[torch.LongTensor] = None,  # If target_mask is not given, we need this to calculate it
        prefix_blocked: Optional[bool] = None,  # Optional, allows to override config value 'on the fly'
        unconditional: Optional[bool] = None,  # Optional, allows to override config value 'on the fly'
    ) -> EchoModelOutput:
        # Override config params if specified.
        blocked = self.prefix_blocked if prefix_blocked is None else prefix_blocked

        # Since 'forward' corresponds to a single model pass, it does not run the
        # unconditional branch by default. The exception is the teacher during CFG
        # distilation, which has to show both branches of the mixture at once.
        guided = self.unconditional if unconditional is None else unconditional

        # Target mask calculation.
        # Uses the provided mask, or recalculates it from a scratch using span lengths if not given.
        if target_mask is None and (blocked or guided):
            # We need span lengths to calculate the target mask
            if prefix_lengths is None or target_lengths is None:
                raise ValueError(
                    "prefix blocking and CFG need 'target_mask', or both "
                    "'prefix_lengths' and 'target_lengths'"
                )

            target_mask = build_target_mask(
                prefix_lengths, target_lengths, input_ids.device
            )

        # Prefix blocking fixup
        # The undelying base model's 'forward' has no idea about whether prefix blocking is applied or not -
        # so it has to be communicated via a proper attention mask, passed to the underlying LLM backbone.
        if blocked:
            # Let's construct the mask first if not given.
            if attention_mask is None:
                attention_mask = build_dense_bidirectional_mask(
                    torch.ones_like(audio_mask, dtype=torch.bool)
                )

            attention_mask = apply_prefix_blocking_4d(attention_mask, target_mask)
            document_ids = None

        outputs = super().forward(
            input_ids=input_ids,
            audio_mask=audio_mask,
            labels=labels,
            attention_mask=attention_mask,
            document_ids=document_ids,
            position_ids=position_ids,
        )

        # Except for the teacher-CFG mode, we finish after a single pass.
        if not guided:
            return EchoModelOutput(loss=outputs.loss, logits=outputs.logits)

        starts, lengths = target_spans(target_mask)
        cond_logits, target_valid = gather_target(outputs.logits, starts, lengths)

        return EchoModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            cond_logits=cond_logits,
            uncond_logits=self._uncond_logits(
                input_ids, starts, lengths, target_valid
            ),
            target_valid=target_valid,
        )

    def _uncond_logits(
        self,
        input_ids: torch.LongTensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Logits for the target span alone, as the decoder's uncond rows."""
        target_ids, _ = gather_target(input_ids, starts, lengths)
        target_ids = torch.where(
            valid.unsqueeze(1), target_ids, AUDIO_MASK_ID
        )
        positions = (
            torch.arange(target_ids.size(-1), device=input_ids.device)
            .unsqueeze(0)
            .expand(target_ids.size(0), -1)
        )
        return (
            super()
            .forward(
                input_ids=target_ids,
                audio_mask=valid,
                attention_mask=build_dense_bidirectional_mask(valid),
                position_ids=positions,
            )
            .logits
        )

    # Full Diffiusion Process
    # Completes N passes (steps) via the base 'generate()' method
    def generate(
        self,
        text: Union[str, List[str]],
        *args: Any,
        guidance_scale: Optional[float] = None,
        prefix_blocked: Optional[bool] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        gen_config = (
            generation_config
            if generation_config is not None
            else self.echo_config.default_generation_config(**kwargs)
        )
        if guidance_scale is not None:
            gen_config.guidance_scale = guidance_scale
        if prefix_blocked is not None:
            gen_config.prefix_blocked = prefix_blocked

        # While the base decoder drives, it builds the blocked attention mask and
        # the guided rows itself, and calls 'forward' without any spans - so for
        # the length of the call, 'forward' has to be a plain single pass.
        blocked, unconditional = self.prefix_blocked, self.unconditional
        self.prefix_blocked, self.unconditional = False, False

        try:
            return super().generate(
                text, *args, generation_config=gen_config, **kwargs
            )
        finally:
            self.prefix_blocked, self.unconditional = blocked, unconditional
