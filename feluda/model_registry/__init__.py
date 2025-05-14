"""
Model registry for Feluda.

This module provides a model registry for managing machine learning models.
"""

from feluda.model_registry.registry import (
    Model,
    ModelRegistry,
    ModelVersion,
    get_model_registry,
)
from feluda.model_registry.storage import (
    ModelStorage,
    ModelStorageBackend,
    FileModelStorage,
    S3ModelStorage,
    get_model_storage,
)
from feluda.model_registry.metadata import (
    ModelMetadata,
    MetadataStore,
    SQLiteMetadataStore,
    get_metadata_store,
)

__all__ = [
    "FileModelStorage",
    "Model",
    "ModelMetadata",
    "ModelRegistry",
    "ModelStorage",
    "ModelStorageBackend",
    "ModelVersion",
    "MetadataStore",
    "S3ModelStorage",
    "SQLiteMetadataStore",
    "get_metadata_store",
    "get_model_registry",
    "get_model_storage",
]
