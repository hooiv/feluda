"""
Metrics for experiment tracking.

This module provides metrics for experiment tracking.
"""

import abc
import enum
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class MetricType(str, enum.Enum):
    """Enum for metric types."""
    
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


class Metric(BaseModel):
    """
    Metric.
    
    This class represents a metric in the experiment tracking system.
    """
    
    experiment_id: str = Field(..., description="The experiment ID")
    run_id: str = Field(..., description="The run ID")
    key: str = Field(..., description="The metric key")
    value: float = Field(..., description="The metric value")
    step: Optional[int] = Field(None, description="The step number")
    timestamp: float = Field(..., description="The timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metric to a dictionary.
        
        Returns:
            A dictionary representation of the metric.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metric":
        """
        Create a metric from a dictionary.
        
        Args:
            data: The dictionary to create the metric from.
            
        Returns:
            A metric.
        """
        return cls(**data)


class MetricLogger:
    """
    Metric logger.
    
    This class is responsible for logging metrics.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the metric logger.
        
        Args:
            db_path: The path to the SQLite database.
        """
        config = get_config()
        self.db_path = db_path or config.metric_logger_db or "experiments/metrics.db"
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
            
            # Create the metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id TEXT,
                    run_id TEXT,
                    key TEXT,
                    value REAL,
                    step INTEGER,
                    timestamp REAL,
                    PRIMARY KEY (experiment_id, run_id, key, step)
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id ON metrics (experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics (key)")
            
            # Commit the changes
            conn.commit()
    
    def log_metric(
        self,
        experiment_id: str,
        run_id: str,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a metric.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            key: The metric key.
            value: The metric value.
            step: The step number.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Create the metric
            import time
            
            metric = Metric(
                experiment_id=experiment_id,
                run_id=run_id,
                key=key,
                value=value,
                step=step,
                timestamp=time.time(),
            )
            
            # Save the metric
            conn.execute(
                """
                INSERT OR REPLACE INTO metrics (experiment_id, run_id, key, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.experiment_id,
                    metric.run_id,
                    metric.key,
                    metric.value,
                    metric.step,
                    metric.timestamp,
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_metrics(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        key: Optional[str] = None,
    ) -> List[Metric]:
        """
        Get metrics.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            key: The metric key.
            
        Returns:
            A list of metrics.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Build the query
            query = "SELECT * FROM metrics"
            params = []
            
            conditions = []
            
            if experiment_id:
                conditions.append("experiment_id = ?")
                params.append(experiment_id)
            
            if run_id:
                conditions.append("run_id = ?")
                params.append(run_id)
            
            if key:
                conditions.append("key = ?")
                params.append(key)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp"
            
            # Execute the query
            cursor = conn.execute(query, params)
            
            # Create the metrics
            return [
                Metric(
                    experiment_id=row["experiment_id"],
                    run_id=row["run_id"],
                    key=row["key"],
                    value=row["value"],
                    step=row["step"],
                    timestamp=row["timestamp"],
                )
                for row in cursor.fetchall()
            ]
    
    def get_metric_history(
        self,
        experiment_id: str,
        run_id: str,
        key: str,
    ) -> List[Tuple[Optional[int], float]]:
        """
        Get the history of a metric.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            key: The metric key.
            
        Returns:
            A list of (step, value) tuples.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Execute the query
            cursor = conn.execute(
                """
                SELECT step, value
                FROM metrics
                WHERE experiment_id = ? AND run_id = ? AND key = ?
                ORDER BY step
                """,
                (experiment_id, run_id, key),
            )
            
            # Create the history
            return [(row["step"], row["value"]) for row in cursor.fetchall()]
    
    def get_latest_metrics(
        self,
        experiment_id: str,
        run_id: str,
    ) -> Dict[str, float]:
        """
        Get the latest metrics for a run.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            
        Returns:
            A dictionary mapping metric keys to values.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Execute the query
            cursor = conn.execute(
                """
                SELECT key, value
                FROM metrics
                WHERE experiment_id = ? AND run_id = ?
                GROUP BY key
                HAVING timestamp = MAX(timestamp)
                """,
                (experiment_id, run_id),
            )
            
            # Create the metrics
            return {row["key"]: row["value"] for row in cursor.fetchall()}
    
    def delete_metrics(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """
        Delete metrics.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            key: The metric key.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Build the query
            query = "DELETE FROM metrics"
            params = []
            
            conditions = []
            
            if experiment_id:
                conditions.append("experiment_id = ?")
                params.append(experiment_id)
            
            if run_id:
                conditions.append("run_id = ?")
                params.append(run_id)
            
            if key:
                conditions.append("key = ?")
                params.append(key)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Execute the query
            conn.execute(query, params)
            
            # Commit the changes
            conn.commit()


# Global metric logger instance
_metric_logger = None
_metric_logger_lock = threading.RLock()


def get_metric_logger() -> MetricLogger:
    """
    Get the global metric logger instance.
    
    Returns:
        The global metric logger instance.
    """
    global _metric_logger
    
    with _metric_logger_lock:
        if _metric_logger is None:
            _metric_logger = MetricLogger()
        
        return _metric_logger
