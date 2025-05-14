"""
Artifacts for experiment tracking.

This module provides artifacts for experiment tracking.
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
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class Artifact(BaseModel):
    """
    Artifact.
    
    This class represents an artifact in the experiment tracking system.
    """
    
    name: str = Field(..., description="The artifact name")
    path: str = Field(..., description="The path to the artifact file")
    run_id: str = Field(..., description="The run ID")
    experiment_id: str = Field(..., description="The experiment ID")
    created_at: float = Field(..., description="The creation timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the artifact to a dictionary.
        
        Returns:
            A dictionary representation of the artifact.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """
        Create an artifact from a dictionary.
        
        Args:
            data: The dictionary to create the artifact from.
            
        Returns:
            An artifact.
        """
        return cls(**data)


class ArtifactStorageBackend(abc.ABC):
    """
    Base class for artifact storage backends.
    
    This class defines the interface for artifact storage backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def save_artifact(self, artifact: Artifact) -> str:
        """
        Save an artifact.
        
        Args:
            artifact: The artifact to save.
            
        Returns:
            The path to the saved artifact.
        """
        pass
    
    @abc.abstractmethod
    def get_artifact(self, path: str) -> str:
        """
        Get an artifact.
        
        Args:
            path: The path to the artifact.
            
        Returns:
            The path to the artifact file.
        """
        pass
    
    @abc.abstractmethod
    def delete_artifact(self, path: str) -> None:
        """
        Delete an artifact.
        
        Args:
            path: The path to the artifact.
        """
        pass


class FileArtifactStorage(ArtifactStorageBackend):
    """
    File artifact storage.
    
    This class implements an artifact storage backend that stores artifacts in files.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize a file artifact storage.
        
        Args:
            base_dir: The base directory.
        """
        self.base_dir = base_dir
        
        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
    
    def _get_artifact_dir(self, experiment_id: str, run_id: str) -> str:
        """
        Get the directory for an artifact.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            
        Returns:
            The artifact directory.
        """
        return os.path.join(self.base_dir, experiment_id, run_id)
    
    def _get_artifact_path(self, experiment_id: str, run_id: str, name: str) -> str:
        """
        Get the path for an artifact.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            name: The artifact name.
            
        Returns:
            The artifact path.
        """
        return os.path.join(self._get_artifact_dir(experiment_id, run_id), name)
    
    def save_artifact(self, artifact: Artifact) -> str:
        """
        Save an artifact.
        
        Args:
            artifact: The artifact to save.
            
        Returns:
            The path to the saved artifact.
        """
        # Create the artifact directory
        artifact_dir = self._get_artifact_dir(artifact.experiment_id, artifact.run_id)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Get the artifact path
        artifact_path = self._get_artifact_path(artifact.experiment_id, artifact.run_id, artifact.name)
        
        # Copy the artifact file
        shutil.copy2(artifact.path, artifact_path)
        
        return artifact_path
    
    def get_artifact(self, path: str) -> str:
        """
        Get an artifact.
        
        Args:
            path: The path to the artifact.
            
        Returns:
            The path to the artifact file.
        """
        # Check if the artifact file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Artifact file not found: {path}")
        
        return path
    
    def delete_artifact(self, path: str) -> None:
        """
        Delete an artifact.
        
        Args:
            path: The path to the artifact.
        """
        # Check if the artifact file exists
        if os.path.isfile(path):
            # Delete the artifact file
            os.remove(path)


class S3ArtifactStorage(ArtifactStorageBackend):
    """
    S3 artifact storage.
    
    This class implements an artifact storage backend that stores artifacts in S3.
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
        Initialize an S3 artifact storage.
        
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
    
    def _get_artifact_key(self, experiment_id: str, run_id: str, name: str) -> str:
        """
        Get the S3 key for an artifact.
        
        Args:
            experiment_id: The experiment ID.
            run_id: The run ID.
            name: The artifact name.
            
        Returns:
            The S3 key.
        """
        return f"{self.prefix}{experiment_id}/{run_id}/{name}"
    
    def save_artifact(self, artifact: Artifact) -> str:
        """
        Save an artifact.
        
        Args:
            artifact: The artifact to save.
            
        Returns:
            The path to the saved artifact.
        """
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Get the S3 key
        key = self._get_artifact_key(artifact.experiment_id, artifact.run_id, artifact.name)
        
        # Upload the artifact file
        s3_client.upload_file(artifact.path, self.bucket, key)
        
        return f"s3://{self.bucket}/{key}"
    
    def get_artifact(self, path: str) -> str:
        """
        Get an artifact.
        
        Args:
            path: The path to the artifact.
            
        Returns:
            The path to the artifact file.
        """
        # Parse the S3 path
        if not path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {path}")
        
        path = path[5:]
        bucket, key = path.split("/", 1)
        
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Download the artifact file to a temporary location
        import tempfile
        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        try:
            s3_client.download_file(bucket, key, temp_file.name)
        except Exception as e:
            os.unlink(temp_file.name)
            raise FileNotFoundError(f"Artifact file not found: {path}") from e
        
        return temp_file.name
    
    def delete_artifact(self, path: str) -> None:
        """
        Delete an artifact.
        
        Args:
            path: The path to the artifact.
        """
        # Parse the S3 path
        if not path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {path}")
        
        path = path[5:]
        bucket, key = path.split("/", 1)
        
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Delete the artifact file
        try:
            s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            log.error(f"Error deleting artifact: {e}")


class ArtifactStorage:
    """
    Artifact storage.
    
    This class is responsible for storing and loading artifacts.
    """
    
    def __init__(self, backend: Optional[ArtifactStorageBackend] = None):
        """
        Initialize the artifact storage.
        
        Args:
            backend: The artifact storage backend.
        """
        config = get_config()
        
        if backend:
            self.backend = backend
        elif config.artifact_storage_type == "s3":
            self.backend = S3ArtifactStorage(
                bucket=config.artifact_storage_bucket,
                prefix=config.artifact_storage_prefix,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                region_name=config.aws_region_name,
            )
        else:
            self.backend = FileArtifactStorage(
                base_dir=config.artifact_storage_dir or "experiments/artifacts",
            )
    
    def save_artifact(self, artifact: Artifact) -> str:
        """
        Save an artifact.
        
        Args:
            artifact: The artifact to save.
            
        Returns:
            The path to the saved artifact.
        """
        return self.backend.save_artifact(artifact)
    
    def get_artifact(self, path: str) -> str:
        """
        Get an artifact.
        
        Args:
            path: The path to the artifact.
            
        Returns:
            The path to the artifact file.
        """
        return self.backend.get_artifact(path)
    
    def delete_artifact(self, path: str) -> None:
        """
        Delete an artifact.
        
        Args:
            path: The path to the artifact.
        """
        self.backend.delete_artifact(path)


# Global artifact storage instance
_artifact_storage = None
_artifact_storage_lock = threading.RLock()


def get_artifact_storage() -> ArtifactStorage:
    """
    Get the global artifact storage instance.
    
    Returns:
        The global artifact storage instance.
    """
    global _artifact_storage
    
    with _artifact_storage_lock:
        if _artifact_storage is None:
            _artifact_storage = ArtifactStorage()
        
        return _artifact_storage
