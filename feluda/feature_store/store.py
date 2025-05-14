"""
Feature store for Feluda.

This module provides a feature store for machine learning features.
"""

import abc
import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.feature_store.storage import FeatureStorage, get_feature_storage
from feluda.observability import get_logger

log = get_logger(__name__)


class FeatureType(str, enum.Enum):
    """Enum for feature types."""
    
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class Feature(BaseModel):
    """
    Feature.
    
    This class represents a feature in the feature store.
    """
    
    name: str = Field(..., description="The feature name")
    description: Optional[str] = Field(None, description="The feature description")
    type: FeatureType = Field(..., description="The feature type")
    group: str = Field(..., description="The feature group")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="The feature metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the feature to a dictionary.
        
        Returns:
            A dictionary representation of the feature.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        """
        Create a feature from a dictionary.
        
        Args:
            data: The dictionary to create the feature from.
            
        Returns:
            A feature.
        """
        return cls(**data)


class FeatureGroup(BaseModel):
    """
    Feature group.
    
    This class represents a group of features in the feature store.
    """
    
    name: str = Field(..., description="The feature group name")
    description: Optional[str] = Field(None, description="The feature group description")
    features: Dict[str, Feature] = Field(default_factory=dict, description="The features in the group")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="The feature group metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the feature group to a dictionary.
        
        Returns:
            A dictionary representation of the feature group.
        """
        return {
            "name": self.name,
            "description": self.description,
            "features": {
                name: feature.to_dict()
                for name, feature in self.features.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureGroup":
        """
        Create a feature group from a dictionary.
        
        Args:
            data: The dictionary to create the feature group from.
            
        Returns:
            A feature group.
        """
        features = {
            name: Feature.from_dict(feature)
            for name, feature in data.get("features", {}).items()
        }
        
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            features=features,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
        )
    
    def get_feature(self, name: str) -> Optional[Feature]:
        """
        Get a feature by name.
        
        Args:
            name: The feature name.
            
        Returns:
            The feature, or None if the feature is not found.
        """
        return self.features.get(name)
    
    def add_feature(self, feature: Feature) -> None:
        """
        Add a feature to the group.
        
        Args:
            feature: The feature to add.
        """
        self.features[feature.name] = feature
        self.updated_at = time.time()
    
    def remove_feature(self, name: str) -> bool:
        """
        Remove a feature from the group.
        
        Args:
            name: The feature name.
            
        Returns:
            True if the feature was removed, False otherwise.
        """
        if name in self.features:
            del self.features[name]
            self.updated_at = time.time()
            return True
        
        return False


class FeatureStore:
    """
    Feature store.
    
    This class is responsible for managing features.
    """
    
    def __init__(self, storage: Optional[FeatureStorage] = None):
        """
        Initialize the feature store.
        
        Args:
            storage: The feature storage.
        """
        self.storage = storage or get_feature_storage()
        self.groups: Dict[str, FeatureGroup] = {}
        self.lock = threading.RLock()
        
        # Load feature groups from the storage
        self._load_groups()
    
    def _load_groups(self) -> None:
        """
        Load feature groups from the storage.
        """
        try:
            # Get all feature groups
            groups = self.storage.get_groups()
            
            # Store the feature groups
            self.groups = groups
        
        except Exception as e:
            log.error(f"Error loading feature groups: {e}")
    
    def create_group(self, name: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> FeatureGroup:
        """
        Create a feature group.
        
        Args:
            name: The feature group name.
            description: The feature group description.
            metadata: The feature group metadata.
            
        Returns:
            The created feature group.
        """
        with self.lock:
            # Check if the group already exists
            if name in self.groups:
                return self.groups[name]
            
            # Create the feature group
            now = time.time()
            
            group = FeatureGroup(
                name=name,
                description=description,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            
            # Store the feature group
            self.groups[name] = group
            self.storage.save_group(group)
            
            return group
    
    def get_group(self, name: str) -> Optional[FeatureGroup]:
        """
        Get a feature group by name.
        
        Args:
            name: The feature group name.
            
        Returns:
            The feature group, or None if the group is not found.
        """
        with self.lock:
            return self.groups.get(name)
    
    def get_groups(self) -> Dict[str, FeatureGroup]:
        """
        Get all feature groups.
        
        Returns:
            A dictionary mapping feature group names to feature groups.
        """
        with self.lock:
            return self.groups.copy()
    
    def delete_group(self, name: str) -> bool:
        """
        Delete a feature group.
        
        Args:
            name: The feature group name.
            
        Returns:
            True if the group was deleted, False otherwise.
        """
        with self.lock:
            # Check if the group exists
            if name not in self.groups:
                return False
            
            # Delete the feature group
            del self.groups[name]
            self.storage.delete_group(name)
            
            return True
    
    def create_feature(
        self,
        group_name: str,
        name: str,
        type: FeatureType,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Feature]:
        """
        Create a feature.
        
        Args:
            group_name: The feature group name.
            name: The feature name.
            type: The feature type.
            description: The feature description.
            metadata: The feature metadata.
            
        Returns:
            The created feature, or None if the group is not found.
        """
        with self.lock:
            # Get the feature group
            group = self.get_group(group_name)
            
            if not group:
                return None
            
            # Check if the feature already exists
            feature = group.get_feature(name)
            
            if feature:
                return feature
            
            # Create the feature
            now = time.time()
            
            feature = Feature(
                name=name,
                description=description,
                type=type,
                group=group_name,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            
            # Add the feature to the group
            group.add_feature(feature)
            
            # Save the feature group
            self.storage.save_group(group)
            
            return feature
    
    def get_feature(self, group_name: str, name: str) -> Optional[Feature]:
        """
        Get a feature by name.
        
        Args:
            group_name: The feature group name.
            name: The feature name.
            
        Returns:
            The feature, or None if the feature is not found.
        """
        with self.lock:
            # Get the feature group
            group = self.get_group(group_name)
            
            if not group:
                return None
            
            # Get the feature
            return group.get_feature(name)
    
    def delete_feature(self, group_name: str, name: str) -> bool:
        """
        Delete a feature.
        
        Args:
            group_name: The feature group name.
            name: The feature name.
            
        Returns:
            True if the feature was deleted, False otherwise.
        """
        with self.lock:
            # Get the feature group
            group = self.get_group(group_name)
            
            if not group:
                return False
            
            # Remove the feature from the group
            if not group.remove_feature(name):
                return False
            
            # Save the feature group
            self.storage.save_group(group)
            
            # Delete the feature values
            self.storage.delete_feature_values(group_name, name)
            
            return True
    
    def get_feature_values(
        self,
        group_name: str,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            entity_ids: The entity IDs. If None, get values for all entities.
            
        Returns:
            A dictionary mapping entity IDs to feature values.
        """
        with self.lock:
            # Get the feature
            feature = self.get_feature(group_name, feature_name)
            
            if not feature:
                return {}
            
            # Get the feature values
            return self.storage.get_feature_values(group_name, feature_name, entity_ids)
    
    def set_feature_values(
        self,
        group_name: str,
        feature_name: str,
        values: Dict[str, Any],
    ) -> bool:
        """
        Set feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            values: A dictionary mapping entity IDs to feature values.
            
        Returns:
            True if the values were set, False otherwise.
        """
        with self.lock:
            # Get the feature
            feature = self.get_feature(group_name, feature_name)
            
            if not feature:
                return False
            
            # Set the feature values
            self.storage.set_feature_values(group_name, feature_name, values)
            
            return True
    
    def get_entity_features(
        self,
        group_name: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get features for an entity.
        
        Args:
            group_name: The feature group name.
            entity_id: The entity ID.
            feature_names: The feature names. If None, get all features.
            
        Returns:
            A dictionary mapping feature names to feature values.
        """
        with self.lock:
            # Get the feature group
            group = self.get_group(group_name)
            
            if not group:
                return {}
            
            # Get the feature names
            if feature_names is None:
                feature_names = list(group.features.keys())
            
            # Get the feature values
            result = {}
            
            for feature_name in feature_names:
                # Get the feature
                feature = group.get_feature(feature_name)
                
                if not feature:
                    continue
                
                # Get the feature values
                values = self.storage.get_feature_values(group_name, feature_name, [entity_id])
                
                if entity_id in values:
                    result[feature_name] = values[entity_id]
            
            return result
    
    def get_entity_dataframe(
        self,
        group_name: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get a DataFrame of features for entities.
        
        Args:
            group_name: The feature group name.
            entity_ids: The entity IDs.
            feature_names: The feature names. If None, get all features.
            
        Returns:
            A DataFrame of features for entities.
        """
        with self.lock:
            # Get the feature group
            group = self.get_group(group_name)
            
            if not group:
                return pd.DataFrame()
            
            # Get the feature names
            if feature_names is None:
                feature_names = list(group.features.keys())
            
            # Get the feature values
            data = []
            
            for entity_id in entity_ids:
                # Get the entity features
                features = self.get_entity_features(group_name, entity_id, feature_names)
                
                # Add the entity ID
                features["entity_id"] = entity_id
                
                # Add the features to the data
                data.append(features)
            
            # Create the DataFrame
            return pd.DataFrame(data)


# Global feature store instance
_feature_store = None
_feature_store_lock = threading.RLock()


def get_feature_store() -> FeatureStore:
    """
    Get the global feature store instance.
    
    Returns:
        The global feature store instance.
    """
    global _feature_store
    
    with _feature_store_lock:
        if _feature_store is None:
            _feature_store = FeatureStore()
        
        return _feature_store
