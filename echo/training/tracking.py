import os
import subprocess

from pathlib import Path
from typing import Any, Dict

from echo.training.config import EchoTrainingConfig
from echo.utils.config_utils import read_json
from echo.utils.distributed_utils import world

# Kept beside the checkpoints, so a resumed run reports to the run it continues.
RUN_ID_FILE = "mlflow_run_id"

# Single step states sent to MLflow: the first, middle and last main-path passes of s8, and the polish pass.
# The full table, per language too, is still printed to the log.
LOGGED_STATES = ("s8_f0", "s8_f3", "s8_f7", "s8_polish")


class MLflowTracker:
    """
    Reports a training run to MLflow: its settings once, and its curves as it goes.
    """

    def __init__(self, config: EchoTrainingConfig) -> None:
        # Imported here, so only a tracked run needs mlflow installed.
        import mlflow

        self.mlflow = mlflow
        self.loss = config.loss
        tracking = config.tracking

        # MLflow reads its basic auth credentials from the environment only.
        if tracking.username is not None:
            os.environ["MLFLOW_TRACKING_USERNAME"] = tracking.username
        if tracking.password is not None:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking.password

        mlflow.set_tracking_uri(tracking.tracking_uri)

        run_id = Path(config.checkpoint.output_dir) / RUN_ID_FILE

        if config.checkpoint.resume is not None and run_id.is_file():
            mlflow.start_run(run_id=run_id.read_text().strip())
            return

        run = mlflow.start_run(
            experiment_id=tracking.experiment_id,
            run_name=config.meta.name,
            description=config.meta.notes,
            tags={"version": str(config.meta.version), "world_size": str(world()[1]), **_commit()},
        )
        run_id.parent.mkdir(parents=True, exist_ok=True)
        run_id.write_text(run.info.run_id)

        # The credentials stay out of what the run page shows.
        settings = config.to_dict()
        del settings["tracking"]

        params = _flatten(settings)
        for role in ("student", "teacher"):
            path = getattr(config.models, role)
            if path is not None:
                params.update(_flatten(read_json(path), f"{role}."))
                mlflow.log_artifact(path, "configs")

        mlflow.log_params(params)
        mlflow.log_dict(settings, "configs/training.json")

    def log_train(self, reported: Dict[str, float], step: int) -> None:
        metrics = {
            "train/loss": reported["loss"],
            "learning_rate": reported["learning_rate"],
            "epoch": reported["epoch"],
        }
        if any(path.ce > 0.0 for path in self.loss.paths):
            metrics["train/ce"] = reported["ce"]
        if any(path.kl > 0.0 for path in self.loss.paths):
            metrics["train/kl"] = reported["kl"]

        self._log(metrics, step)

    def log_eval(self, name: str, results: Dict[Any, Any], step: int, with_teacher: bool) -> None:
        self._log(eval_metrics(name, results, with_teacher), step)

    def close(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status)

    def _log(self, metrics: Dict[str, float], step: int) -> None:
        # Sent in the background, so a slow server never holds a training step up.
        self.mlflow.log_metrics(metrics, step=step, synchronous=False)


def eval_metrics(
    name: str,                  # The evaluated set: 'dev' or 'test'
    results: Dict[Any, Any],    # What evaluate_single_step returned
    with_teacher: bool,
) -> Dict[str, float]:
    """
    The CE gap to the teacher (or the reference) at the LOGGED_STATES; without one to compare to, the plain CE.
    """

    term = "ce_delta" if with_teacher else "ce"

    return {f"{name}/{term}/{state}": results[state][term] for state in LOGGED_STATES}


def _flatten(settings: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}

    for key, value in settings.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value

    return flat


def _commit() -> Dict[str, str]:
    """
    The commit the run was started from, when it was started from a checkout.
    """

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)

    return {"mlflow.source.git.commit": head.stdout.strip()} if head.returncode == 0 else {}
