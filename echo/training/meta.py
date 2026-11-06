import torch

from typing import Dict

from echo.training.types import MixedPrecision

# ---------- Precision ----------

# dtype aliases
MIXED_PRECISION_DTYPES: Dict[MixedPrecision, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
