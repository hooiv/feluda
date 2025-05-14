"""
Model metadata for Feluda.

This module provides metadata for machine learning models.
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


class ModelMetadata(BaseModel):
    """
    Model metadata.
    
    This class represents metadata for a model.
    """
    
    framework: Optional[str] = Field(None, description="The model framework")
    framework_version: Optional[str] = Field(None, description="The model framework version")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="The model input schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="The model output schema")
    metrics: Dict[str, float] = Field(default_factory=dict, description="The model metrics")
    tags: List[str] = Field(default_factory=list, description="The model tags")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model metadata to a dictionary.
        
        Returns:
            A dictionary representation of the model metadata.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """
        Create model metadata from a dictionary.
        
        Args:
            data: The dictionary to create the model metadata from.
            
        Returns:
            Model metadata.
        """
        return cls(**data)


class MetadataStore(abc.ABC):
    """
    Base class for metadata stores.
    
    This class defines the interface for metadata stores.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def save_model(self, model: Any) -> None:
        """
        Save a model.
        
        Args:
            model: The model to save.
        """
        pass
    
    @abc.abstractmethod
    def get_model(self, name: str) -> Optional[Any]:
        """
        Get a model by name.
        
        Args:
            name: The model name.
            
        Returns:
            The model, or None if the model is not found.
        """
        pass
    
    @abc.abstractmethod
    def get_models(self) -> Dict[str, Any]:
        """
        Get all models.
        
        Returns:
            A dictionary mapping model names to models.
        """
        pass
    
    @abc.abstractmethod
    def delete_model(self, name: str) -> None:
        """
        Delete a model.
        
        Args:
            name: The model name.
        """
        pass


class SQLiteMetadataStore(MetadataStore):
    """
    SQLite metadata store.
    
    This class implements a metadata store that stores metadata in SQLite.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize a SQLite metadata store.
        
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
            
            # Create the models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    created_at REAL,
                    updated_at REAL,
                    data TEXT
                )
            """)
            
            # Commit the changes
            conn.commit()
    
    def save_model(self, model: Any) -> None:
        """
        Save a model.
        
        Args:
            model: The model to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Convert the model to a dictionary
            model_dict = model.to_dict()
            
            # Save the model
            conn.execute(
                """
                INSERT OR REPLACE INTO models (name, description, created_at, updated_at, data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model.name,
                    model.description,
                    model.created_at,
                    model.updated_at,
                    json.dumps(model_dict),
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_model(self, name: str) -> Optional[Any]:
        """
        Get a model by name.
        
        Args:
            name: The model name.
            
        Returns:
            The model, or None if the model is not found.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the model
            cursor = conn.execute(
                "SELECT data FROM models WHERE name = ?",
                (name,),
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse the model data
            model_dict = json.loads(row["data"])
            
            # Import the Model class
            from feluda.model_registry.registry import Model
            
            # Create the model
            return Model.from_dict(model_dict)
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get all models.
        
        Returns:
            A dictionary mapping model names to models.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get all models
            cursor = conn.execute("SELECT name, data FROM models")
            
            # Import the Model class
            from feluda.model_registry.registry import Model
            
            # Create the models
            return {
                row["name"]: Model.from_dict(json.loads(row["data"]))
                for row in cursor.fetchall()
            }
    
    def delete_model(self, name: str) -> None:
        """
        Delete a model.
        
        Args:
            name: The model name.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the model
            conn.execute(
                "DELETE FROM models WHERE name = ?",
                (name,),
            )
            
            # Commit the changes
            conn.commit()


# Global metadata store instance
_metadata_store = None
_metadata_store_lock = threading.RLock()


def get_metadata_store() -> MetadataStore:
    """
    Get the global metadata store instance.
    
    Returns:
        The global metadata store instance.
    """
    global _metadata_store
    
    with _metadata_store_lock:
        if _metadata_store is None:
            config = get_config()
            
            # Create the metadata store
            _metadata_store = SQLiteMetadataStore(
                db_path=config.model_metadata_db or "models/metadata.db",
            )
        
        return _metadata_store
