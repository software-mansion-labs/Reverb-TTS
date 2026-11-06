"""
The MLflow run of one RL training run: its recipe once, its curves as it trains, and its evaluations later, with the
recipe's credentials (or, when it leaves them out, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD).
"""

import os

from pathlib import Path
from typing import Any, Dict, Optional

from echo.rl.config import RLConfig
from echo.training.tracking import RUN_ID_FILE, _commit, _flatten
from echo.utils.config_utils import write_json

RECIPE_FILE = "recipe.json"   # The recipe a run trained with, kept beside it


class Tracker:
    def __init__(self, config: RLConfig, run: Path, params: Optional[Dict[str, Any]] = None,
                 run_id: Optional[str] = None) -> None:
        # Imported here, so only a tracked run needs mlflow installed.
        import mlflow

        self.mlflow = mlflow
        tracking = config.tracking
        if tracking.username is not None:
            os.environ["MLFLOW_TRACKING_USERNAME"] = tracking.username
        if tracking.password is not None:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking.password
        mlflow.set_tracking_uri(tracking.tracking_uri)
        if run_id is not None:
            mlflow.start_run(run_id=run_id)
            return

        started = mlflow.start_run(experiment_id=tracking.experiment_id, run_name=run.name, description=config.notes,
                                   tags={"stage": "rl", **_commit()})
        (run / RUN_ID_FILE).write_text(started.info.run_id)
        # The credentials stay out of what the run page shows.
        settings = config.to_dict()
        del settings["tracking"]
        mlflow.log_params(_flatten({**settings, "run": params or {}}))

    @classmethod
    def start(cls, config: RLConfig, run: Path, params: Dict[str, Any]) -> Optional["Tracker"]:
        """
        A new run's tracker, or None when the recipe doesn't track. The recipe is kept in the run either way.
        """

        run.mkdir(parents=True, exist_ok=True)
        write_json(run / RECIPE_FILE, config.to_dict())
        return cls(config, run, params) if config.tracking.enabled else None

    @classmethod
    def resume(cls, run: Path) -> Optional["Tracker"]:
        """
        The tracker of a trained run, or None when it wasn't tracked.
        """

        if not (run / RUN_ID_FILE).is_file():
            return None
        return cls(RLConfig.from_json(run / RECIPE_FILE), run, run_id=(run / RUN_ID_FILE).read_text().strip())

    def log(self, metrics: Dict[str, float], step: int) -> None:
        # Sent in the background, so a slow server never holds a training step up.
        self.mlflow.log_metrics(metrics, step=step, synchronous=False)

    def close(self) -> None:
        self.mlflow.end_run()
