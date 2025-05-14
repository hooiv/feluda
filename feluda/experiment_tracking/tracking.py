"""
Experiment tracking for Feluda.

This module provides experiment tracking for machine learning experiments.
"""

import abc
import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.experiment_tracking.artifacts import Artifact, get_artifact_storage
from feluda.experiment_tracking.metrics import MetricLogger, get_metric_logger
from feluda.experiment_tracking.storage import ExperimentStorage, get_experiment_storage
from feluda.observability import get_logger

log = get_logger(__name__)


class ExperimentStatus(str, enum.Enum):
    """Enum for experiment status."""
    
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ExperimentRun(BaseModel):
    """
    Experiment run.
    
    This class represents a run of an experiment.
    """
    
    id: str = Field(..., description="The run ID")
    experiment_id: str = Field(..., description="The experiment ID")
    name: str = Field(..., description="The run name")
    description: Optional[str] = Field(None, description="The run description")
    status: ExperimentStatus = Field(..., description="The run status")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="The run parameters")
    metrics: Dict[str, float] = Field(default_factory=dict, description="The run metrics")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="The run artifacts")
    tags: List[str] = Field(default_factory=list, description="The run tags")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    started_at: Optional[float] = Field(None, description="The start timestamp")
    completed_at: Optional[float] = Field(None, description="The completion timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment run to a dictionary.
        
        Returns:
            A dictionary representation of the experiment run.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """
        Create an experiment run from a dictionary.
        
        Args:
            data: The dictionary to create the experiment run from.
            
        Returns:
            An experiment run.
        """
        return cls(**data)


class Experiment(BaseModel):
    """
    Experiment.
    
    This class represents an experiment in the experiment tracking system.
    """
    
    id: str = Field(..., description="The experiment ID")
    name: str = Field(..., description="The experiment name")
    description: Optional[str] = Field(None, description="The experiment description")
    runs: Dict[str, ExperimentRun] = Field(default_factory=dict, description="The experiment runs")
    tags: List[str] = Field(default_factory=list, description="The experiment tags")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment to a dictionary.
        
        Returns:
            A dictionary representation of the experiment.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "runs": {
                run_id: run.to_dict()
                for run_id, run in self.runs.items()
            },
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """
        Create an experiment from a dictionary.
        
        Args:
            data: The dictionary to create the experiment from.
            
        Returns:
            An experiment.
        """
        runs = {
            run_id: ExperimentRun.from_dict(run)
            for run_id, run in data.get("runs", {}).items()
        }
        
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            runs=runs,
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get a run by ID.
        
        Args:
            run_id: The run ID.
            
        Returns:
            The run, or None if the run is not found.
        """
        return self.runs.get(run_id)
    
    def add_run(self, run: ExperimentRun) -> None:
        """
        Add a run to the experiment.
        
        Args:
            run: The run to add.
        """
        self.runs[run.id] = run
        self.updated_at = time.time()
    
    def remove_run(self, run_id: str) -> bool:
        """
        Remove a run from the experiment.
        
        Args:
            run_id: The run ID.
            
        Returns:
            True if the run was removed, False otherwise.
        """
        if run_id in self.runs:
            del self.runs[run_id]
            self.updated_at = time.time()
            return True
        
        return False


class ExperimentTracker:
    """
    Experiment tracker.
    
    This class is responsible for tracking experiments.
    """
    
    def __init__(
        self,
        storage: Optional[ExperimentStorage] = None,
        metric_logger: Optional[MetricLogger] = None,
        artifact_storage: Optional[Any] = None,
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            storage: The experiment storage.
            metric_logger: The metric logger.
            artifact_storage: The artifact storage.
        """
        self.storage = storage or get_experiment_storage()
        self.metric_logger = metric_logger or get_metric_logger()
        self.artifact_storage = artifact_storage or get_artifact_storage()
        self.experiments: Dict[str, Experiment] = {}
        self.active_runs: Dict[str, ExperimentRun] = {}
        self.lock = threading.RLock()
        
        # Load experiments from the storage
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """
        Load experiments from the storage.
        """
        try:
            # Get all experiments
            experiments = self.storage.get_experiments()
            
            # Store the experiments
            self.experiments = experiments
        
        except Exception as e:
            log.error(f"Error loading experiments: {e}")
    
    def create_experiment(self, name: str, description: Optional[str] = None, tags: Optional[List[str]] = None) -> Experiment:
        """
        Create an experiment.
        
        Args:
            name: The experiment name.
            description: The experiment description.
            tags: The experiment tags.
            
        Returns:
            The created experiment.
        """
        with self.lock:
            # Check if an experiment with the same name already exists
            for experiment in self.experiments.values():
                if experiment.name == name:
                    return experiment
            
            # Create the experiment
            now = time.time()
            
            experiment = Experiment(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                tags=tags or [],
                created_at=now,
                updated_at=now,
            )
            
            # Store the experiment
            self.experiments[experiment.id] = experiment
            self.storage.save_experiment(experiment)
            
            return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: The experiment ID.
            
        Returns:
            The experiment, or None if the experiment is not found.
        """
        with self.lock:
            return self.experiments.get(experiment_id)
    
    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """
        Get an experiment by name.
        
        Args:
            name: The experiment name.
            
        Returns:
            The experiment, or None if the experiment is not found.
        """
        with self.lock:
            for experiment in self.experiments.values():
                if experiment.name == name:
                    return experiment
            
            return None
    
    def get_experiments(self) -> Dict[str, Experiment]:
        """
        Get all experiments.
        
        Returns:
            A dictionary mapping experiment IDs to experiments.
        """
        with self.lock:
            return self.experiments.copy()
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: The experiment ID.
            
        Returns:
            True if the experiment was deleted, False otherwise.
        """
        with self.lock:
            # Check if the experiment exists
            experiment = self.get_experiment(experiment_id)
            
            if not experiment:
                return False
            
            # Delete the experiment
            del self.experiments[experiment_id]
            self.storage.delete_experiment(experiment_id)
            
            # Delete all runs
            for run_id in list(experiment.runs.keys()):
                self.delete_run(experiment_id, run_id)
            
            return True
    
    def start_run(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ExperimentRun]:
        """
        Start a run.
        
        Args:
            experiment_id: The experiment ID.
            name: The run name.
            description: The run description.
            parameters: The run parameters.
            tags: The run tags.
            
        Returns:
            The created run, or None if the experiment is not found.
        """
        with self.lock:
            # Get the experiment
            experiment = self.get_experiment(experiment_id)
            
            if not experiment:
                return None
            
            # Create the run
            now = time.time()
            
            run = ExperimentRun(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                name=name or f"Run {len(experiment.runs) + 1}",
                description=description,
                status=ExperimentStatus.RUNNING,
                parameters=parameters or {},
                tags=tags or [],
                created_at=now,
                updated_at=now,
                started_at=now,
            )
            
            # Add the run to the experiment
            experiment.add_run(run)
            
            # Store the run
            self.active_runs[run.id] = run
            self.storage.save_experiment(experiment)
            
            return run
    
    def get_run(self, experiment_id: str, run_id: str) -> Optional[ExperimentRun]:
        """
        Get a run by ID.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            
        Returns:
            The run, or None if the run is not found.
        """
        with self.lock:
            # Get the experiment
            experiment = self.get_experiment(experiment_id)
            
            if not experiment:
                return None
            
            # Get the run
            return experiment.get_run(run_id)
    
    def get_active_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get an active run by ID.
        
        Args:
            run_id: The run ID.
            
        Returns:
            The run, or None if the run is not found or not active.
        """
        with self.lock:
            return self.active_runs.get(run_id)
    
    def get_runs(self, experiment_id: str) -> Dict[str, ExperimentRun]:
        """
        Get all runs for an experiment.
        
        Args:
            experiment_id: The experiment ID.
            
        Returns:
            A dictionary mapping run IDs to runs.
        """
        with self.lock:
            # Get the experiment
            experiment = self.get_experiment(experiment_id)
            
            if not experiment:
                return {}
            
            # Get the runs
            return experiment.runs.copy()
    
    def delete_run(self, experiment_id: str, run_id: str) -> bool:
        """
        Delete a run.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            
        Returns:
            True if the run was deleted, False otherwise.
        """
        with self.lock:
            # Get the experiment
            experiment = self.get_experiment(experiment_id)
            
            if not experiment:
                return False
            
            # Get the run
            run = experiment.get_run(run_id)
            
            if not run:
                return False
            
            # Remove the run from the active runs
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            
            # Remove the run from the experiment
            experiment.remove_run(run_id)
            
            # Save the experiment
            self.storage.save_experiment(experiment)
            
            # Delete the run artifacts
            for artifact_name, artifact_path in run.artifacts.items():
                self.artifact_storage.delete_artifact(artifact_path)
            
            return True
    
    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> bool:
        """
        Log a metric.
        
        Args:
            run_id: The run ID.
            key: The metric key.
            value: The metric value.
            step: The step number.
            
        Returns:
            True if the metric was logged, False otherwise.
        """
        with self.lock:
            # Get the run
            run = self.get_active_run(run_id)
            
            if not run:
                return False
            
            # Log the metric
            self.metric_logger.log_metric(run.experiment_id, run_id, key, value, step)
            
            # Update the run
            run.metrics[key] = value
            run.updated_at = time.time()
            
            # Save the experiment
            experiment = self.get_experiment(run.experiment_id)
            
            if experiment:
                self.storage.save_experiment(experiment)
            
            return True
    
    def log_artifact(self, run_id: str, name: str, path: str) -> Optional[str]:
        """
        Log an artifact.
        
        Args:
            run_id: The run ID.
            name: The artifact name.
            path: The path to the artifact file.
            
        Returns:
            The artifact path, or None if the run is not found.
        """
        with self.lock:
            # Get the run
            run = self.get_active_run(run_id)
            
            if not run:
                return None
            
            # Create the artifact
            artifact = Artifact(
                name=name,
                path=path,
                run_id=run_id,
                experiment_id=run.experiment_id,
                created_at=time.time(),
            )
            
            # Save the artifact
            artifact_path = self.artifact_storage.save_artifact(artifact)
            
            # Update the run
            run.artifacts[name] = artifact_path
            run.updated_at = time.time()
            
            # Save the experiment
            experiment = self.get_experiment(run.experiment_id)
            
            if experiment:
                self.storage.save_experiment(experiment)
            
            return artifact_path
    
    def get_artifact(self, run_id: str, name: str) -> Optional[str]:
        """
        Get an artifact.
        
        Args:
            run_id: The run ID.
            name: The artifact name.
            
        Returns:
            The path to the artifact file, or None if the artifact is not found.
        """
        with self.lock:
            # Get the run
            run = None
            
            # Check active runs first
            run = self.get_active_run(run_id)
            
            if not run:
                # Check all experiments
                for experiment in self.experiments.values():
                    run = experiment.get_run(run_id)
                    
                    if run:
                        break
            
            if not run:
                return None
            
            # Get the artifact path
            artifact_path = run.artifacts.get(name)
            
            if not artifact_path:
                return None
            
            # Get the artifact
            return self.artifact_storage.get_artifact(artifact_path)
    
    def end_run(self, run_id: str, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> bool:
        """
        End a run.
        
        Args:
            run_id: The run ID.
            status: The run status.
            
        Returns:
            True if the run was ended, False otherwise.
        """
        with self.lock:
            # Get the run
            run = self.get_active_run(run_id)
            
            if not run:
                return False
            
            # Update the run
            run.status = status
            run.completed_at = time.time()
            run.updated_at = time.time()
            
            # Remove the run from the active runs
            del self.active_runs[run_id]
            
            # Save the experiment
            experiment = self.get_experiment(run.experiment_id)
            
            if experiment:
                self.storage.save_experiment(experiment)
            
            return True


# Global experiment tracker instance
_experiment_tracker = None
_experiment_tracker_lock = threading.RLock()


def get_experiment_tracker() -> ExperimentTracker:
    """
    Get the global experiment tracker instance.
    
    Returns:
        The global experiment tracker instance.
    """
    global _experiment_tracker
    
    with _experiment_tracker_lock:
        if _experiment_tracker is None:
            _experiment_tracker = ExperimentTracker()
        
        return _experiment_tracker
