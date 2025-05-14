"""
Data Models Module

This module defines versioned Pydantic models for data structures used throughout Feluda.
These models provide validation, serialization, and documentation for data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_serializer, model_validator


class VersionedModel(BaseModel):
    """
    Base class for versioned models.
    
    All models that need versioning should inherit from this class.
    It provides a version field and methods for handling version compatibility.
    """
    
    model_version: str = Field(
        default="1.0.0",
        description="The version of the model schema."
    )
    
    @classmethod
    def get_latest_version(cls) -> str:
        """
        Get the latest version of the model.
        
        Returns:
            The latest version string.
        """
        return "1.0.0"  # This should be updated when new versions are added
    
    @model_validator(mode="before")
    @classmethod
    def check_version_compatibility(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the model version is compatible with the current implementation.
        
        Args:
            data: The data to validate.
            
        Returns:
            The validated data.
            
        Raises:
            ValueError: If the version is incompatible.
        """
        if isinstance(data, dict) and "model_version" in data:
            version = data["model_version"]
            latest_version = cls.get_latest_version()
            
            # Simple version check for now
            # In a real implementation, this would be more sophisticated
            if version != latest_version:
                # For now, we'll just warn about version mismatch
                # In the future, we might implement migration logic here
                import warnings
                warnings.warn(
                    f"Model version mismatch: got {version}, latest is {latest_version}. "
                    f"This may cause compatibility issues."
                )
        
        return data


class MediaType(str, Enum):
    """Enum for media types."""
    
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class MediaMetadata(VersionedModel):
    """
    Model for media metadata.
    
    This model represents metadata for various types of media.
    """
    
    media_type: MediaType = Field(
        default=MediaType.UNKNOWN,
        description="The type of media."
    )
    
    mime_type: Optional[str] = Field(
        default=None,
        description="The MIME type of the media."
    )
    
    size_bytes: Optional[int] = Field(
        default=None,
        description="The size of the media in bytes."
    )
    
    width: Optional[int] = Field(
        default=None,
        description="The width of the media in pixels (for images and videos)."
    )
    
    height: Optional[int] = Field(
        default=None,
        description="The height of the media in pixels (for images and videos)."
    )
    
    duration_seconds: Optional[float] = Field(
        default=None,
        description="The duration of the media in seconds (for videos and audio)."
    )
    
    created_at: Optional[datetime] = Field(
        default=None,
        description="The creation timestamp of the media."
    )
    
    modified_at: Optional[datetime] = Field(
        default=None,
        description="The last modification timestamp of the media."
    )
    
    additional_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata specific to the media type."
    )
    
    @field_serializer('created_at', 'modified_at')
    def serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format."""
        return dt.isoformat() if dt else None


class MediaContent(VersionedModel):
    """
    Model for media content.
    
    This model represents the content of various types of media.
    """
    
    metadata: MediaMetadata = Field(
        default_factory=MediaMetadata,
        description="Metadata about the media."
    )
    
    content_uri: Optional[str] = Field(
        default=None,
        description="URI pointing to the media content."
    )
    
    content_data: Optional[Any] = Field(
        default=None,
        description="The actual media content data (if available)."
    )
    
    @model_validator(mode="after")
    def check_content_availability(self) -> "MediaContent":
        """
        Check if either content_uri or content_data is provided.
        
        Returns:
            The validated model.
            
        Raises:
            ValueError: If neither content_uri nor content_data is provided.
        """
        if self.content_uri is None and self.content_data is None:
            raise ValueError("Either content_uri or content_data must be provided.")
        return self


class EmbeddingModel(VersionedModel):
    """
    Model for embeddings.
    
    This model represents vector embeddings generated by various models.
    """
    
    embedding: List[float] = Field(
        ...,
        description="The embedding vector."
    )
    
    model_name: str = Field(
        ...,
        description="The name of the model used to generate the embedding."
    )
    
    model_version: str = Field(
        ...,
        description="The version of the model used to generate the embedding."
    )
    
    dimensions: int = Field(
        ...,
        description="The number of dimensions in the embedding."
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when the embedding was created."
    )
    
    @model_validator(mode="after")
    def check_dimensions(self) -> "EmbeddingModel":
        """
        Check if the dimensions match the length of the embedding.
        
        Returns:
            The validated model.
            
        Raises:
            ValueError: If the dimensions don't match the embedding length.
        """
        if len(self.embedding) != self.dimensions:
            raise ValueError(
                f"Embedding dimensions ({self.dimensions}) don't match "
                f"the actual length of the embedding ({len(self.embedding)})."
            )
        return self
    
    @field_serializer('created_at')
    def serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format."""
        return dt.isoformat()


class OperatorResult(VersionedModel):
    """
    Model for operator results.
    
    This model represents the result of an operator execution.
    """
    
    operator_name: str = Field(
        ...,
        description="The name of the operator that produced the result."
    )
    
    operator_version: str = Field(
        ...,
        description="The version of the operator that produced the result."
    )
    
    result_data: Any = Field(
        ...,
        description="The result data produced by the operator."
    )
    
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="The execution time in milliseconds."
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when the result was created."
    )
    
    additional_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information about the result."
    )
    
    @field_serializer('created_at')
    def serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format."""
        return dt.isoformat()


class OperatorConfig(VersionedModel):
    """
    Model for operator configuration.
    
    This model represents the configuration for an operator.
    """
    
    operator_name: str = Field(
        ...,
        description="The name of the operator."
    )
    
    operator_type: str = Field(
        ...,
        description="The type of the operator."
    )
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the operator."
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether the operator is enabled."
    )
    
    priority: int = Field(
        default=0,
        description="The priority of the operator (higher values mean higher priority)."
    )
    
    timeout_seconds: Optional[float] = Field(
        default=None,
        description="The timeout for operator execution in seconds."
    )
    
    retry_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for retrying failed operations."
    )
