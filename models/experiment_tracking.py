"""
Experiment Tracking and Model Registry
MLflow integration for reproducible research and model versioning
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from infrastructure.logger import quant_logger

logger = quant_logger.get_logger("experiment_tracking")


class ExperimentTracker:
    """
    Experiment tracking system (MLflow-compatible)
    Tracks parameters, metrics, artifacts, and models
    """

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_id: Optional[str] = None
        self.current_experiment_id: Optional[str] = None

    def create_experiment(self, name: str, description: str = "") -> str:
        """
        Create a new experiment

        Args:
            name: Experiment name
            description: Experiment description

        Returns:
            experiment_id
        """
        experiment_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        experiment_dir = self.experiments_dir / experiment_id

        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }

        with open(experiment_dir / "experiment.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.current_experiment_id = experiment_id

        logger.info(f"Created experiment: {name} (ID: {experiment_id})")

        return experiment_id

    def start_run(
        self,
        run_name: str,
        experiment_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new run

        Args:
            run_name: Run name
            experiment_id: Parent experiment ID
            tags: Run tags

        Returns:
            run_id
        """
        if experiment_id is None:
            experiment_id = self.current_experiment_id

        if experiment_id is None:
            raise ValueError("No experiment selected. Call create_experiment first.")

        # Generate run ID
        timestamp = datetime.now().isoformat()
        run_id = hashlib.sha256(f"{run_name}_{timestamp}".encode()).hexdigest()[:16]

        run_dir = self.experiments_dir / experiment_id / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize run metadata
        run_metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "experiment_id": experiment_id,
            "start_time": timestamp,
            "tags": tags or {},
            "status": "running",
            "parameters": {},
            "metrics": {},
            "artifacts": [],
        }

        with open(run_dir / "run.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        self.current_run_id = run_id

        logger.info(f"Started run: {run_name} (ID: {run_id})")

        return run_id

    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None):
        """
        Log parameters for a run

        Args:
            params: Dictionary of parameters
            run_id: Run ID (uses current run if None)
        """
        if run_id is None:
            run_id = self.current_run_id

        if run_id is None:
            raise ValueError("No active run. Call start_run first.")

        run_dir = self._get_run_dir(run_id)

        # Update run metadata
        with open(run_dir / "run.json", "r") as f:
            run_metadata = json.load(f)

        run_metadata["parameters"].update(params)

        with open(run_dir / "run.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        logger.debug(f"Logged {len(params)} parameters to run {run_id}")

    def log_metrics(
        self, metrics: Dict[str, float], step: int = 0, run_id: Optional[str] = None
    ):
        """
        Log metrics for a run

        Args:
            metrics: Dictionary of metrics
            step: Step number (for tracking over time)
            run_id: Run ID
        """
        if run_id is None:
            run_id = self.current_run_id

        if run_id is None:
            raise ValueError("No active run")

        run_dir = self._get_run_dir(run_id)

        # Append metrics to metrics file
        metrics_file = run_dir / "metrics.jsonl"

        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": metrics,
        }

        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")

        # Update run metadata with latest metrics
        with open(run_dir / "run.json", "r") as f:
            run_metadata = json.load(f)

        run_metadata["metrics"].update(metrics)

        with open(run_dir / "run.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        logger.debug(f"Logged {len(metrics)} metrics to run {run_id}")

    def log_artifact(
        self, artifact_path: Path, artifact_name: str, run_id: Optional[str] = None
    ):
        """
        Log an artifact (file) for a run

        Args:
            artifact_path: Path to artifact file
            artifact_name: Name to save artifact as
            run_id: Run ID
        """
        if run_id is None:
            run_id = self.current_run_id

        run_dir = self._get_run_dir(run_id)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Copy artifact
        import shutil

        shutil.copy(artifact_path, artifacts_dir / artifact_name)

        # Update run metadata
        with open(run_dir / "run.json", "r") as f:
            run_metadata = json.load(f)

        run_metadata["artifacts"].append(artifact_name)

        with open(run_dir / "run.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        logger.info(f"Logged artifact: {artifact_name}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ):
        """
        Log a model (pickling)

        Args:
            model: Model object to save
            model_name: Model name
            metadata: Model metadata
            run_id: Run ID
        """
        if run_id is None:
            run_id = self.current_run_id

        run_dir = self._get_run_dir(run_id)
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Save model
        model_path = models_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save model metadata
        model_metadata = {
            "model_name": model_name,
            "saved_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        with open(models_dir / f"{model_name}_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        logger.info(f"Logged model: {model_name}")

    def end_run(self, status: str = "completed", run_id: Optional[str] = None):
        """
        End a run

        Args:
            status: Run status ('completed', 'failed', 'killed')
            run_id: Run ID
        """
        if run_id is None:
            run_id = self.current_run_id

        run_dir = self._get_run_dir(run_id)

        with open(run_dir / "run.json", "r") as f:
            run_metadata = json.load(f)

        run_metadata["status"] = status
        run_metadata["end_time"] = datetime.now().isoformat()

        with open(run_dir / "run.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        logger.info(f"Ended run {run_id} with status: {status}")

        self.current_run_id = None

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run metadata"""
        run_dir = self._get_run_dir(run_id)

        with open(run_dir / "run.json", "r") as f:
            return json.load(f)

    def list_runs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List all runs in an experiment"""
        runs_dir = self.experiments_dir / experiment_id / "runs"

        if not runs_dir.exists():
            return []

        runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                with open(run_dir / "run.json", "r") as f:
                    runs.append(json.load(f))

        return runs

    def compare_runs(
        self, run_ids: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple runs

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare (None = all)

        Returns:
            DataFrame comparing runs
        """
        comparison_data = []

        for run_id in run_ids:
            run_metadata = self.get_run(run_id)

            row = {
                "run_id": run_id,
                "run_name": run_metadata["run_name"],
                "status": run_metadata["status"],
            }

            # Add parameters
            for param, value in run_metadata["parameters"].items():
                row[f"param_{param}"] = value

            # Add metrics
            for metric, value in run_metadata["metrics"].items():
                if metrics is None or metric in metrics:
                    row[f"metric_{metric}"] = value

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def _get_run_dir(self, run_id: str) -> Path:
        """Get run directory"""
        # Search all experiments
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            run_dir = exp_dir / "runs" / run_id

            if run_dir.exists():
                return run_dir

        raise ValueError(f"Run {run_id} not found")


class ModelRegistry:
    """
    Model registry for versioning and deployment
    """

    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        model: Any,
        metadata: Dict[str, Any],
        stage: str = "staging",  # 'staging', 'production', 'archived'
    ) -> str:
        """
        Register a model in the registry

        Args:
            model_name: Model name
            model: Model object
            metadata: Model metadata (performance, data version, etc.)
            stage: Deployment stage

        Returns:
            version_id
        """
        # Create model directory
        model_dir = self.registry_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Generate version
        existing_versions = list(model_dir.glob("v*"))
        version_num = len(existing_versions) + 1
        version_id = f"v{version_num}"

        version_dir = model_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(version_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        full_metadata = {
            "model_name": model_name,
            "version": version_id,
            "registered_at": datetime.now().isoformat(),
            "stage": stage,
            "metadata": metadata,
        }

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.info(f"Registered model {model_name} {version_id} (stage: {stage})")

        return version_id

    def promote_model(self, model_name: str, version_id: str, to_stage: str):
        """
        Promote model to a different stage

        Args:
            model_name: Model name
            version_id: Version ID
            to_stage: Target stage ('production', 'staging', 'archived')
        """
        version_dir = self.registry_dir / model_name / version_id

        with open(version_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        metadata["stage"] = to_stage
        metadata["promoted_at"] = datetime.now().isoformat()

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Promoted {model_name} {version_id} to {to_stage}")

    def load_model(
        self,
        model_name: str,
        version_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from registry

        Args:
            model_name: Model name
            version_id: Specific version (or latest if None)
            stage: Load model from specific stage (e.g., 'production')

        Returns:
            (model, metadata)
        """
        model_dir = self.registry_dir / model_name

        if not model_dir.exists():
            raise ValueError(f"Model {model_name} not found in registry")

        # Find version
        if version_id is None:
            if stage is not None:
                # Find latest version in stage
                versions = []
                for v_dir in model_dir.glob("v*"):
                    with open(v_dir / "metadata.json", "r") as f:
                        meta = json.load(f)
                        if meta["stage"] == stage:
                            versions.append((v_dir.name, meta["registered_at"]))

                if not versions:
                    raise ValueError(f"No model in stage {stage}")

                # Sort by registration time
                versions.sort(key=lambda x: x[1], reverse=True)
                version_id = versions[0][0]
            else:
                # Latest version
                versions = sorted(model_dir.glob("v*"), key=lambda x: int(x.name[1:]))
                version_id = versions[-1].name

        version_dir = model_dir / version_id

        # Load model
        with open(version_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load metadata
        with open(version_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded model {model_name} {version_id}")

        return model, metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []

        for model_dir in self.registry_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for version_dir in model_dir.glob("v*"):
                with open(version_dir / "metadata.json", "r") as f:
                    metadata = json.load(f)
                    models.append(metadata)

        return models
