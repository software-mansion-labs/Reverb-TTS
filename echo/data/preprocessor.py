import random
import torch

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Tuple, Union

from echo.data.constants import (
    DEFAULT_TOKENIZER,
    EMPTY,
    FRAME_RATE,
    PROMPT_TEMPLATE,
    TEXT_TEMPLATE,
)
from echo.masking.audio_masking import build_main_stage_audio_mask, build_polish_stage_audio_mask
from echo.masking.constants import DEFAULT_POLISH_SHARE


class EchoPreprocessor:
    """
    Turns raw :class:`EchoDataset` samples into real model inputs.

    This operator sits between the dataset and the data loader's collate.
    It does a few things:
    1. Tokenizes the transcript
    2. Builds the prompt (with PROMPT_TEMPLATE)
    3. Randomly splits the audio part into prefix and target, with a respect to defined rules
    4. Randomly masks the audio target part with MASK_TOKEN(s), at a main-path or a polish state of the sampler
    """

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizerBase] = DEFAULT_TOKENIZER,
        prefix_min_dur: float = 1.0,     # Seconds of audio the prefix always keeps
        prefix_min_ratio: float = 0.0,   # Lower bound of the prefix ratio draw
        prefix_max_ratio: float = 0.5,   # Upper bound of the prefix ratio draw
        prefix_clamp: bool = False,      # True: clamp short draws up to prefix_min_dur, instead of drawing above it
        polish_share: float = DEFAULT_POLISH_SHARE,  # Share of samples masked at the polish state, the rest at a main-path one
        masking_kwargs: Optional[Dict[str, Any]] = None,  # Arguments of the main-path mask builder - allows passing them from the training config
        drop_prefix_ratio: float = 0.0,  # Share of a batch left without any prefix
    ) -> None:
        # Initialize tokenizer
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer)
            if isinstance(tokenizer, str)
            else tokenizer
        )

        self.prefix_min_frames = int(prefix_min_dur * FRAME_RATE)
        self.prefix_min_ratio = prefix_min_ratio
        self.prefix_max_ratio = prefix_max_ratio
        self.prefix_clamp = prefix_clamp

        self.polish_share = polish_share
        self.masking_kwargs = masking_kwargs or {}
        self.drop_prefix_ratio = drop_prefix_ratio

        # A precomputed prompt set.
        # Since number of languages is limited, the prompts are very repetetive.
        self._prompts: Dict[str, torch.Tensor] = {}

    def __call__(
        self,
        samples: List[Dict[str, Any]],
        disable_prefix_drop: bool = False,   # Skips the unconditional-branch samples
        disable_masking: bool = False,       # Leaves the target audio intact
    ) -> List[Dict[str, Any]]:
        # Some samples lose their prefix, so that the model also learns the
        # unconditional branch it needs for CFG. 
        count = 0 if disable_prefix_drop else round(self.drop_prefix_ratio * len(samples))
        dropped = set(random.sample(range(len(samples)), count))

        return [
            self.process(sample, i in dropped, disable_masking)
            for i, sample in enumerate(samples)
        ]

    # Main processing operator.
    def process(
        self,
        sample: Dict[str, Any],
        drop_prefix: bool = False,
        disable_masking: bool = False,
    ) -> Dict[str, Any]:
        # The unconditional case, or simply "drop prefix" case, uses neither text nor prompt in the input.
        prompt_tokens = EMPTY if drop_prefix else self.prompt_tokens(sample["language"])
        text_tokens = EMPTY if drop_prefix else self.text_tokens(sample["text"])

        audio_tokens = sample["audio_tokens"]
        prefix_frames, target_frames = (
            (0, audio_tokens.size(-1)) if drop_prefix else self.split_audio(audio_tokens)
        )

        # The mask is drawn here, but not applied at the preprocessing stage here.
        # This allows the Trainer to construct the labels.
        polish = not disable_masking and random.random() < self.polish_share
        mask = (
            torch.zeros_like(audio_tokens, dtype=torch.bool)
            if disable_masking
            else self.draw_mask(audio_tokens, prefix_frames, polish)
        )

        prefix_length = len(prompt_tokens) + len(text_tokens) + prefix_frames

        return {
            "prompt_tokens": prompt_tokens,
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens,
            "mask": mask,
            "polish": polish,
            "prefix_length": prefix_length,
            "target_length": target_frames,
            "length": prefix_length + target_frames,
            "speaker_id": sample["speaker_id"],
            "language": sample["language"],
        }

    def prompt_tokens(self, language: str) -> torch.Tensor:
        """
        Builds the prompt by - or looks it up if already cached.
        """

        if language not in self._prompts:
            prompt = PROMPT_TEMPLATE.format(language=language)
            self._prompts[language] = self._encode(prompt)

        return self._prompts[language]

    def text_tokens(self, text: str) -> torch.Tensor:
        """
        Tokenizes the text and adds special text opening and closing marks.
        """
        return self._encode(TEXT_TEMPLATE.format(text=text))

    # Suboperator - audio splitting
    def split_audio(
        self, 
        audio_tokens: torch.Tensor
    ) -> Tuple[int, int]:
        """
        Splits the audio into a prefix the model may condition on and a target
        it has to produce.

        The ratio is drawn per sample, so the same utterance is split differently every time it is seen.
        """

        total_frames = audio_tokens.size(-1)

        # Drawn above the minimum rather than clamped to it, so no share of the prefixes piles up at exactly the
        # minimum (a third of them at 1.5 s, with the training set's lengths). An utterance too short to fit the
        # minimum below the upper bound falls back to the clamp.
        low = self.prefix_min_ratio
        if not self.prefix_clamp:
            low = max(low, self.prefix_min_frames / max(total_frames, 1))
        ratio = random.uniform(low, self.prefix_max_ratio) if low <= self.prefix_max_ratio else self.prefix_min_ratio

        # Short prefixes carry too little of the voice to condition on, and a
        # prefix can never outgrow the utterance it is taken from.
        prefix_frames = max(int(ratio * total_frames), self.prefix_min_frames)
        prefix_frames = min(prefix_frames, total_frames)

        return prefix_frames, total_frames - prefix_frames

    # Suboperator - audio target masking
    def draw_mask(
        self,
        audio_tokens: torch.Tensor,
        prefix_frames: int,   # Allows to localize the target span
        polish: bool,         # The polish state, instead of a main-path one
    ) -> torch.Tensor:
        """
        Marks the target frames the model will have to restore, leaving the
        prefix alone. The rates are drawn per sample.
        """

        if polish:
            return build_polish_stage_audio_mask(audio_tokens, prefix_frames)

        return build_main_stage_audio_mask(audio_tokens, prefix_frames, **self.masking_kwargs)

    def _encode(self, text: str) -> torch.Tensor:
        """
        A helper function, used as a tokenizer's PyTorch wrapper.
        """

        return torch.tensor(self.tokenizer(text).input_ids, dtype=torch.long)
