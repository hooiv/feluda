"""
Experiment storage for Feluda.

This module provides storage for experiment tracking.
"""

import abc
import enum
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class ExperimentStorageBackend(abc.ABC):
    """
    Base class for experiment storage backends.
    
    This class defines the interface for experiment storage backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def save_experiment(self, experiment: Any) -> None:
        """
        Save an experiment.
        
        Args:
            experiment: The experiment to save.
        """
        pass
    
    @abc.abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[Any]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: The experiment ID.
            
        Returns:
            The experiment, or None if the experiment is not found.
        """
        pass
    
    @abc.abstractmethod
    def get_experiments(self) -> Dict[str, Any]:
        """
        Get all experiments.
        
        Returns:
            A dictionary mapping experiment IDs to experiments.
        """
        pass
    
    @abc.abstractmethod
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment.
        
        Args:
            experiment_id: The experiment ID.
        """
        pass


class SQLiteExperimentStorage(ExperimentStorageBackend):
    """
    SQLite experiment storage.
    
    This class implements an experiment storage backend that stores experiments in SQLite.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize a SQLite experiment storage.
        
        Args:
            db_path: The path to the SQLite database.
        """
        self.db_path = db_path
        self.conn = None
        self.lock = threading.RLock()
        
        # Create the database if it doesn't exist
        self._create_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Returns:
            A connection to the SQLite database.
        """
        if not self.conn:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        
        return self.conn
    
    def _create_database(self) -> None:
        """
        Create the SQLite database.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Create the experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Commit the changes
            conn.commit()
    
    def save_experiment(self, experiment: Any) -> None:
        """
        Save an experiment.
        
        Args:
            experiment: The experiment to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Convert the experiment to a dictionary
            experiment_dict = experiment.to_dict()
            
            # Save the experiment
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (id, data)
                VALUES (?, ?)
                """,
                (
                    experiment.id,
                    json.dumps(experiment_dict),
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_experiment(self, experiment_id: str) -> Optional[Any]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: The experiment ID.
            
        Returns:
            The experiment, or None if the experiment is not found.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the experiment
            cursor = conn.execute(
                "SELECT data FROM experiments WHERE id = ?",
                (experiment_id,),
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse the experiment data
            experiment_dict = json.loads(row["data"])
            
            # Import the Experiment class
            from feluda.experiment_tracking.tracking import Experiment
            
            # Create the experiment
            return Experiment.from_dict(experiment_dict)
    
    def get_experiments(self) -> Dict[str, Any]:
        """
        Get all experiments.
        
        Returns:
            A dictionary mapping experiment IDs to experiments.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get all experiments
            cursor = conn.execute("SELECT id, data FROM experiments")
            
            # Import the Experiment class
            from feluda.experiment_tracking.tracking import Experiment
            
            # Create the experiments
            return {
                row["id"]: Experiment.from_dict(json.loads(row["data"]))
                for row in cursor.fetchall()
            }
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment.
        
        Args:
            experiment_id: The experiment ID.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the experiment
            conn.execute(
                "DELETE FROM experiments WHERE id = ?",
                (experiment_id,),
            )
            
            # Commit the changes
            conn.commit()


class ExperimentStorage:
    """
    Experiment storage.
    
    This class is responsible for storing and loading experiments.
    """
    
    def __init__(self, backend: Optional[ExperimentStorageBackend] = None):
        """
        Initialize the experiment storage.
        
        Args:
            backend: The experiment storage backend.
        """
        config = get_config()
        
        if backend:
            self.backend = backend
        else:
            self.backend = SQLiteExperimentStorage(
                db_path=config.experiment_storage_db or "experiments/experiments.db",
            )
    
    def save_experiment(self, experiment: Any) -> None:
        """
        Save an experiment.
        
        Args:
            experiment: The experiment to save.
        """
        self.backend.save_experiment(experiment)
    
    def get_experiment(self, experiment_id: str) -> Optional[Any]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: The experiment ID.
            
        Returns:
            The experiment, or None if the experiment is not found.
        """
        return self.backend.get_experiment(experiment_id)
    
    def get_experiments(self) -> Dict[str, Any]:
        """
        Get all experiments.
        
        Returns:
            A dictionary mapping experiment IDs to experiments.
        """
        return self.backend.get_experiments()
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment.
        
        Args:
            experiment_id: The experiment ID.
        """
        self.backend.delete_experiment(experiment_id)


# Global experiment storage instance
_experiment_storage = None
_experiment_storage_lock = threading.RLock()


def get_experiment_storage() -> ExperimentStorage:
    """
    Get the global experiment storage instance.
    
    Returns:
        The global experiment storage instance.
    """
    global _experiment_storage
    
    with _experiment_storage_lock:
        if _experiment_storage is None:
            _experiment_storage = ExperimentStorage()
        
        return _experiment_storage
