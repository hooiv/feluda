"""
Model registry for Feluda.

This module provides a model registry for managing machine learning models.
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
from feluda.model_registry.metadata import ModelMetadata, get_metadata_store
from feluda.model_registry.storage import ModelStorage, get_model_storage
from feluda.observability import get_logger

log = get_logger(__name__)


class ModelVersion(BaseModel):
    """
    Model version.
    
    This class represents a version of a model.
    """
    
    version: str = Field(..., description="The model version")
    created_at: float = Field(..., description="The creation timestamp")
    metadata: ModelMetadata = Field(..., description="The model metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model version to a dictionary.
        
        Returns:
            A dictionary representation of the model version.
        """
        return {
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """
        Create a model version from a dictionary.
        
        Args:
            data: The dictionary to create the model version from.
            
        Returns:
            A model version.
        """
        return cls(
            version=data.get("version"),
            created_at=data.get("created_at"),
            metadata=ModelMetadata.from_dict(data.get("metadata", {})),
        )


class Model(BaseModel):
    """
    Model.
    
    This class represents a model in the registry.
    """
    
    name: str = Field(..., description="The model name")
    description: Optional[str] = Field(None, description="The model description")
    versions: Dict[str, ModelVersion] = Field(default_factory=dict, description="The model versions")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            A dictionary representation of the model.
        """
        return {
            "name": self.name,
            "description": self.description,
            "versions": {
                version: model_version.to_dict()
                for version, model_version in self.versions.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """
        Create a model from a dictionary.
        
        Args:
            data: The dictionary to create the model from.
            
        Returns:
            A model.
        """
        versions = {
            version: ModelVersion.from_dict(model_version)
            for version, model_version in data.get("versions", {}).items()
        }
        
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            versions=versions,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """
        Get a model version.
        
        Args:
            version: The model version.
            
        Returns:
            The model version, or None if the version is not found.
        """
        return self.versions.get(version)
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        """
        Get the latest model version.
        
        Returns:
            The latest model version, or None if there are no versions.
        """
        if not self.versions:
            return None
        
        return max(self.versions.values(), key=lambda v: v.created_at)


class ModelRegistry:
    """
    Model registry.
    
    This class is responsible for managing models.
    """
    
    def __init__(self, storage: Optional[ModelStorage] = None):
        """
        Initialize the model registry.
        
        Args:
            storage: The model storage.
        """
        self.storage = storage or get_model_storage()
        self.metadata_store = get_metadata_store()
        self.models: Dict[str, Model] = {}
        self.lock = threading.RLock()
        
        # Load models from the metadata store
        self._load_models()
    
    def _load_models(self) -> None:
        """
        Load models from the metadata store.
        """
        try:
            # Get all models
            models = self.metadata_store.get_models()
            
            # Store the models
            self.models = models
        
        except Exception as e:
            log.error(f"Error loading models: {e}")
    
    def register_model(self, name: str, description: Optional[str] = None) -> Model:
        """
        Register a model.
        
        Args:
            name: The model name.
            description: The model description.
            
        Returns:
            The registered model.
        """
        with self.lock:
            # Check if the model already exists
            if name in self.models:
                return self.models[name]
            
            # Create the model
            now = time.time()
            
            model = Model(
                name=name,
                description=description,
                created_at=now,
                updated_at=now,
            )
            
            # Store the model
            self.models[name] = model
            self.metadata_store.save_model(model)
            
            return model
    
    def get_model(self, name: str) -> Optional[Model]:
        """
        Get a model by name.
        
        Args:
            name: The model name.
            
        Returns:
            The model, or None if the model is not found.
        """
        with self.lock:
            return self.models.get(name)
    
    def get_models(self) -> Dict[str, Model]:
        """
        Get all models.
        
        Returns:
            A dictionary mapping model names to models.
        """
        with self.lock:
            return self.models.copy()
    
    def add_model_version(
        self,
        name: str,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelVersion]:
        """
        Add a model version.
        
        Args:
            name: The model name.
            version: The model version.
            model_path: The path to the model file.
            metadata: The model metadata.
            
        Returns:
            The model version, or None if the model is not found.
        """
        with self.lock:
            # Get the model
            model = self.get_model(name)
            
            if not model:
                return None
            
            # Create the model version
            model_version = ModelVersion(
                version=version,
                created_at=time.time(),
                metadata=ModelMetadata(
                    framework=metadata.get("framework") if metadata else None,
                    framework_version=metadata.get("framework_version") if metadata else None,
                    input_schema=metadata.get("input_schema") if metadata else None,
                    output_schema=metadata.get("output_schema") if metadata else None,
                    metrics=metadata.get("metrics", {}) if metadata else {},
                    tags=metadata.get("tags", []) if metadata else [],
                    custom=metadata.get("custom", {}) if metadata else {},
                ),
            )
            
            # Store the model version
            model.versions[version] = model_version
            model.updated_at = time.time()
            
            # Save the model to the metadata store
            self.metadata_store.save_model(model)
            
            # Save the model file to the storage
            self.storage.save_model(name, version, model_path)
            
            return model_version
    
    def get_model_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """
        Get a model version.
        
        Args:
            name: The model name.
            version: The model version.
            
        Returns:
            The model version, or None if the model or version is not found.
        """
        with self.lock:
            # Get the model
            model = self.get_model(name)
            
            if not model:
                return None
            
            # Get the model version
            return model.get_version(version)
    
    def get_latest_model_version(self, name: str) -> Optional[ModelVersion]:
        """
        Get the latest model version.
        
        Args:
            name: The model name.
            
        Returns:
            The latest model version, or None if the model is not found or has no versions.
        """
        with self.lock:
            # Get the model
            model = self.get_model(name)
            
            if not model:
                return None
            
            # Get the latest model version
            return model.get_latest_version()
    
    def load_model(self, name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Load a model.
        
        Args:
            name: The model name.
            version: The model version. If None, loads the latest version.
            
        Returns:
            The loaded model, or None if the model or version is not found.
        """
        with self.lock:
            # Get the model version
            if version:
                model_version = self.get_model_version(name, version)
            else:
                model_version = self.get_latest_model_version(name)
            
            if not model_version:
                return None
            
            # Load the model from the storage
            return self.storage.load_model(name, model_version.version)
    
    def delete_model_version(self, name: str, version: str) -> bool:
        """
        Delete a model version.
        
        Args:
            name: The model name.
            version: The model version.
            
        Returns:
            True if the model version was deleted, False otherwise.
        """
        with self.lock:
            # Get the model
            model = self.get_model(name)
            
            if not model:
                return False
            
            # Check if the version exists
            if version not in model.versions:
                return False
            
            # Delete the model version
            del model.versions[version]
            model.updated_at = time.time()
            
            # Save the model to the metadata store
            self.metadata_store.save_model(model)
            
            # Delete the model file from the storage
            self.storage.delete_model(name, version)
            
            return True
    
    def delete_model(self, name: str) -> bool:
        """
        Delete a model.
        
        Args:
            name: The model name.
            
        Returns:
            True if the model was deleted, False otherwise.
        """
        with self.lock:
            # Get the model
            model = self.get_model(name)
            
            if not model:
                return False
            
            # Delete all model versions
            for version in list(model.versions.keys()):
                self.delete_model_version(name, version)
            
            # Delete the model
            del self.models[name]
            
            # Delete the model from the metadata store
            self.metadata_store.delete_model(name)
            
            return True


# Global model registry instance
_model_registry = None
_model_registry_lock = threading.RLock()


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        The global model registry instance.
    """
    global _model_registry
    
    with _model_registry_lock:
        if _model_registry is None:
            _model_registry = ModelRegistry()
        
        return _model_registry
