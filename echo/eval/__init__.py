"""Echo: evaluation passes shared by the distillation stages."""

from echo.eval.multi_step import evaluate_multi_step
from echo.eval.single_step import evaluate_single_step

__all__ = ["evaluate_multi_step", "evaluate_single_step"]
