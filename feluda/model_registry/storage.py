"""
Model storage for Feluda.

This module provides storage for machine learning models.
"""

import abc
import enum
import json
import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import boto3
import joblib
import torch
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class ModelStorageBackend(abc.ABC):
    """
    Base class for model storage backends.
    
    This class defines the interface for model storage backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def save_model(self, model_name: str, model_version: str, model_path: str) -> None:
        """
        Save a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            model_path: The path to the model file.
        """
        pass
    
    @abc.abstractmethod
    def load_model(self, model_name: str, model_version: str) -> Any:
        """
        Load a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The loaded model.
        """
        pass
    
    @abc.abstractmethod
    def delete_model(self, model_name: str, model_version: str) -> None:
        """
        Delete a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        pass
    
    @abc.abstractmethod
    def list_models(self) -> List[str]:
        """
        List all models.
        
        Returns:
            A list of model names.
        """
        pass
    
    @abc.abstractmethod
    def list_model_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: The model name.
            
        Returns:
            A list of model versions.
        """
        pass


class FileModelStorage(ModelStorageBackend):
    """
    File model storage.
    
    This class implements a model storage backend that stores models in files.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize a file model storage.
        
        Args:
            base_dir: The base directory.
        """
        self.base_dir = base_dir
        
        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
    
    def _get_model_dir(self, model_name: str) -> str:
        """
        Get the directory for a model.
        
        Args:
            model_name: The model name.
            
        Returns:
            The model directory.
        """
        return os.path.join(self.base_dir, model_name)
    
    def _get_model_version_dir(self, model_name: str, model_version: str) -> str:
        """
        Get the directory for a model version.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The model version directory.
        """
        return os.path.join(self._get_model_dir(model_name), model_version)
    
    def _get_model_file_path(self, model_name: str, model_version: str) -> str:
        """
        Get the file path for a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The model file path.
        """
        return os.path.join(self._get_model_version_dir(model_name, model_version), "model.pkl")
    
    def save_model(self, model_name: str, model_version: str, model_path: str) -> None:
        """
        Save a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            model_path: The path to the model file.
        """
        # Create the model version directory
        model_version_dir = self._get_model_version_dir(model_name, model_version)
        os.makedirs(model_version_dir, exist_ok=True)
        
        # Copy the model file
        model_file_path = self._get_model_file_path(model_name, model_version)
        shutil.copy2(model_path, model_file_path)
    
    def load_model(self, model_name: str, model_version: str) -> Any:
        """
        Load a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The loaded model.
        """
        # Get the model file path
        model_file_path = self._get_model_file_path(model_name, model_version)
        
        # Check if the model file exists
        if not os.path.isfile(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        # Load the model
        try:
            # Try to load with joblib
            return joblib.load(model_file_path)
        except Exception as e:
            try:
                # Try to load with torch
                return torch.load(model_file_path)
            except Exception as e2:
                # Try to load with a custom loader
                raise ValueError(f"Failed to load model: {e}, {e2}")
    
    def delete_model(self, model_name: str, model_version: str) -> None:
        """
        Delete a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        # Get the model version directory
        model_version_dir = self._get_model_version_dir(model_name, model_version)
        
        # Check if the directory exists
        if os.path.isdir(model_version_dir):
            # Delete the directory
            shutil.rmtree(model_version_dir)
    
    def list_models(self) -> List[str]:
        """
        List all models.
        
        Returns:
            A list of model names.
        """
        # Get all directories in the base directory
        return [
            name for name in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, name))
        ]
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: The model name.
            
        Returns:
            A list of model versions.
        """
        # Get the model directory
        model_dir = self._get_model_dir(model_name)
        
        # Check if the directory exists
        if not os.path.isdir(model_dir):
            return []
        
        # Get all directories in the model directory
        return [
            name for name in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, name))
        ]


class S3ModelStorage(ModelStorageBackend):
    """
    S3 model storage.
    
    This class implements a model storage backend that stores models in S3.
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """
        Initialize an S3 model storage.
        
        Args:
            bucket: The S3 bucket.
            prefix: The S3 prefix.
            aws_access_key_id: The AWS access key ID.
            aws_secret_access_key: The AWS secret access key.
            region_name: The AWS region name.
        """
        self.bucket = bucket
        self.prefix = prefix
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.s3_client = None
    
    def _get_s3_client(self) -> boto3.client:
        """
        Get the S3 client.
        
        Returns:
            The S3 client.
        """
        if not self.s3_client:
            # Create the S3 client
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )
        
        return self.s3_client
    
    def _get_model_key(self, model_name: str, model_version: str) -> str:
        """
        Get the S3 key for a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The S3 key.
        """
        return f"{self.prefix}{model_name}/{model_version}/model.pkl"
    
    def save_model(self, model_name: str, model_version: str, model_path: str) -> None:
        """
        Save a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            model_path: The path to the model file.
        """
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Get the S3 key
        key = self._get_model_key(model_name, model_version)
        
        # Upload the model file
        s3_client.upload_file(model_path, self.bucket, key)
    
    def load_model(self, model_name: str, model_version: str) -> Any:
        """
        Load a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The loaded model.
        """
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Get the S3 key
        key = self._get_model_key(model_name, model_version)
        
        # Download the model file to a temporary location
        import tempfile
        
        with tempfile.NamedTemporaryFile() as temp_file:
            try:
                s3_client.download_file(self.bucket, key, temp_file.name)
            except Exception as e:
                raise FileNotFoundError(f"Model file not found: {key}")
            
            # Load the model
            try:
                # Try to load with joblib
                return joblib.load(temp_file.name)
            except Exception as e:
                try:
                    # Try to load with torch
                    return torch.load(temp_file.name)
                except Exception as e2:
                    # Try to load with a custom loader
                    raise ValueError(f"Failed to load model: {e}, {e2}")
    
    def delete_model(self, model_name: str, model_version: str) -> None:
        """
        Delete a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Get the S3 key
        key = self._get_model_key(model_name, model_version)
        
        # Delete the model file
        try:
            s3_client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            log.error(f"Error deleting model: {e}")
    
    def list_models(self) -> List[str]:
        """
        List all models.
        
        Returns:
            A list of model names.
        """
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # List objects with the prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix,
                Delimiter="/",
            )
            
            # Get the common prefixes
            prefixes = response.get("CommonPrefixes", [])
            
            # Extract the model names
            return [
                prefix["Prefix"][len(self.prefix):-1]
                for prefix in prefixes
            ]
        
        except Exception as e:
            log.error(f"Error listing models: {e}")
            return []
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: The model name.
            
        Returns:
            A list of model versions.
        """
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # List objects with the prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"{self.prefix}{model_name}/",
                Delimiter="/",
            )
            
            # Get the common prefixes
            prefixes = response.get("CommonPrefixes", [])
            
            # Extract the model versions
            return [
                prefix["Prefix"].split("/")[-2]
                for prefix in prefixes
            ]
        
        except Exception as e:
            log.error(f"Error listing model versions: {e}")
            return []


class ModelStorage:
    """
    Model storage.
    
    This class is responsible for storing and loading models.
    """
    
    def __init__(self, backend: Optional[ModelStorageBackend] = None):
        """
        Initialize the model storage.
        
        Args:
            backend: The model storage backend.
        """
        config = get_config()
        
        if backend:
            self.backend = backend
        elif config.model_storage_type == "s3":
            self.backend = S3ModelStorage(
                bucket=config.model_storage_bucket,
                prefix=config.model_storage_prefix,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                region_name=config.aws_region_name,
            )
        else:
            self.backend = FileModelStorage(
                base_dir=config.model_storage_dir or "models",
            )
    
    def save_model(self, model_name: str, model_version: str, model_path: str) -> None:
        """
        Save a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            model_path: The path to the model file.
        """
        self.backend.save_model(model_name, model_version, model_path)
    
    def load_model(self, model_name: str, model_version: str) -> Any:
        """
        Load a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The loaded model.
        """
        return self.backend.load_model(model_name, model_version)
    
    def delete_model(self, model_name: str, model_version: str) -> None:
        """
        Delete a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        self.backend.delete_model(model_name, model_version)
    
    def list_models(self) -> List[str]:
        """
        List all models.
        
        Returns:
            A list of model names.
        """
        return self.backend.list_models()
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: The model name.
            
        Returns:
            A list of model versions.
        """
        return self.backend.list_model_versions(model_name)


# Global model storage instance
_model_storage = None
_model_storage_lock = threading.RLock()


def get_model_storage() -> ModelStorage:
    """
    Get the global model storage instance.
    
    Returns:
        The global model storage instance.
    """
    global _model_storage
    
    with _model_storage_lock:
        if _model_storage is None:
            _model_storage = ModelStorage()
        
        return _model_storage
