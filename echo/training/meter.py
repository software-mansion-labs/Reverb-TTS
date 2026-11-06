import time
import torch

from typing import Any, Dict


class Meter:
    """
    Keeps what a training run reports between two log lines.

    Loss terms are added once per micro-batch, so the average is taken over
    those rather than over optimizer steps - with gradient accumulation the two
    differ by exactly that factor.
    """

    def __init__(
        self,
        start_step: int = 0,  # Step the clock starts at - where a resumed run picks up
    ) -> None:
        self.started = time.time()
        self.start_step = start_step
        self.totals: Dict[str, float] = {}
        self.counted = 0

    def add(self, terms: Dict[str, Any]) -> None:
        """
        Records one micro-batch.
        """

        # A term may still be a tensor on the device - reading it back is left
        # to the report, so the loop does not wait for the GPU every step.
        for key, value in terms.items():
            self.totals[key] = self.totals.get(key, 0.0) + value

        self.counted += 1

    def report(self, step: int) -> Dict[str, float]:
        """
        The averages since the last report, and what the run costs so far.

        The accumulators are cleared, while the clock is not: the rate is the
        run's average, which is the number that says when it will finish.
        """

        averages = {
            key: float(total) / max(self.counted, 1)
            for key, total in self.totals.items()
        }

        self.totals = {}
        self.counted = 0

        elapsed = max(time.time() - self.started, 1e-6)
        averages["steps_per_second"] = (step - self.start_step) / elapsed

        if torch.cuda.is_available():
            averages["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30

        return averages
