import csv
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset

from echo.data.misc import StringColumn


class EchoDataset(Dataset):
    """
    A wrapper for a dataset represented by a .csv root file.
    """

    def __init__(
        self,
        manifest: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        manifest = Path(manifest)
        self.manifest = manifest
        self.data_root = Path(data_root) if data_root is not None else manifest.parent

        # Declare data structures for holding the .csv content
        names: List[str] = []
        transcripts: List[str] = []
        speaker_ids: List[str] = []
        languages: List[str] = []
        tokens: List[int] = []

        # Load the .csv content
        with open(manifest, newline="") as f:
            # Quoting stays off: a literal '"' in a transcript would otherwise
            # swallow the following rows.
            for row in csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                names.append(row["name"])
                transcripts.append(row["transcript"])
                speaker_ids.append(row["speaker_id"])
                languages.append(row["language"])
                tokens.append(int(row["tokens"]))

        self.names = StringColumn(names)
        self.transcripts = StringColumn(transcripts)
        self.speaker_ids = StringColumn(speaker_ids)
        self.language_names = sorted(set(languages))
        codes = {language: i for i, language in enumerate(self.language_names)}
        self.language_codes = np.array([codes[item] for item in languages], dtype=np.int8)
        self.tokens = np.array(tokens, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        expected = int(self.tokens[index])
        codec = np.load(self.data_root / self.names[index])
        if codec.shape[1] > expected:
            raise ValueError(
                f"{self.names[index]} holds {codec.shape[1]} frames but the "
                f"manifest claims {expected} tokens - regenerate the manifest"
            )

        return {
            "audio_tokens": torch.from_numpy(codec.astype(np.int64)),
            "text": self.transcripts[index],
            "speaker_id": self.speaker_ids[index],
            "language": self.language_names[self.language_codes[index]],
            "tokens": expected,
        }
