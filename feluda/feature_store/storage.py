"""
Feature storage for Feluda.

This module provides storage for machine learning features.
"""

import abc
import enum
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import redis
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class FeatureStorageBackend(abc.ABC):
    """
    Base class for feature storage backends.
    
    This class defines the interface for feature storage backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def save_group(self, group: Any) -> None:
        """
        Save a feature group.
        
        Args:
            group: The feature group to save.
        """
        pass
    
    @abc.abstractmethod
    def get_group(self, name: str) -> Optional[Any]:
        """
        Get a feature group by name.
        
        Args:
            name: The feature group name.
            
        Returns:
            The feature group, or None if the group is not found.
        """
        pass
    
    @abc.abstractmethod
    def get_groups(self) -> Dict[str, Any]:
        """
        Get all feature groups.
        
        Returns:
            A dictionary mapping feature group names to feature groups.
        """
        pass
    
    @abc.abstractmethod
    def delete_group(self, name: str) -> None:
        """
        Delete a feature group.
        
        Args:
            name: The feature group name.
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    def set_feature_values(
        self,
        group_name: str,
        feature_name: str,
        values: Dict[str, Any],
    ) -> None:
        """
        Set feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            values: A dictionary mapping entity IDs to feature values.
        """
        pass
    
    @abc.abstractmethod
    def delete_feature_values(
        self,
        group_name: str,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Delete feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            entity_ids: The entity IDs. If None, delete values for all entities.
        """
        pass


class SQLiteFeatureStorage(FeatureStorageBackend):
    """
    SQLite feature storage.
    
    This class implements a feature storage backend that stores features in SQLite.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize a SQLite feature storage.
        
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
            
            # Create the feature groups table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_groups (
                    name TEXT PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Create the feature values table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_values (
                    group_name TEXT,
                    feature_name TEXT,
                    entity_id TEXT,
                    value TEXT,
                    PRIMARY KEY (group_name, feature_name, entity_id)
                )
            """)
            
            # Commit the changes
            conn.commit()
    
    def save_group(self, group: Any) -> None:
        """
        Save a feature group.
        
        Args:
            group: The feature group to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Convert the group to a dictionary
            group_dict = group.to_dict()
            
            # Save the group
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_groups (name, data)
                VALUES (?, ?)
                """,
                (
                    group.name,
                    json.dumps(group_dict),
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_group(self, name: str) -> Optional[Any]:
        """
        Get a feature group by name.
        
        Args:
            name: The feature group name.
            
        Returns:
            The feature group, or None if the group is not found.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the group
            cursor = conn.execute(
                "SELECT data FROM feature_groups WHERE name = ?",
                (name,),
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse the group data
            group_dict = json.loads(row["data"])
            
            # Import the FeatureGroup class
            from feluda.feature_store.store import FeatureGroup
            
            # Create the group
            return FeatureGroup.from_dict(group_dict)
    
    def get_groups(self) -> Dict[str, Any]:
        """
        Get all feature groups.
        
        Returns:
            A dictionary mapping feature group names to feature groups.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get all groups
            cursor = conn.execute("SELECT name, data FROM feature_groups")
            
            # Import the FeatureGroup class
            from feluda.feature_store.store import FeatureGroup
            
            # Create the groups
            return {
                row["name"]: FeatureGroup.from_dict(json.loads(row["data"]))
                for row in cursor.fetchall()
            }
    
    def delete_group(self, name: str) -> None:
        """
        Delete a feature group.
        
        Args:
            name: The feature group name.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the group
            conn.execute(
                "DELETE FROM feature_groups WHERE name = ?",
                (name,),
            )
            
            # Delete the feature values
            conn.execute(
                "DELETE FROM feature_values WHERE group_name = ?",
                (name,),
            )
            
            # Commit the changes
            conn.commit()
    
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
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the feature values
            if entity_ids:
                # Get values for specific entities
                placeholders = ", ".join(["?"] * len(entity_ids))
                
                cursor = conn.execute(
                    f"""
                    SELECT entity_id, value
                    FROM feature_values
                    WHERE group_name = ? AND feature_name = ? AND entity_id IN ({placeholders})
                    """,
                    (group_name, feature_name) + tuple(entity_ids),
                )
            else:
                # Get values for all entities
                cursor = conn.execute(
                    """
                    SELECT entity_id, value
                    FROM feature_values
                    WHERE group_name = ? AND feature_name = ?
                    """,
                    (group_name, feature_name),
                )
            
            # Create the result
            return {
                row["entity_id"]: json.loads(row["value"])
                for row in cursor.fetchall()
            }
    
    def set_feature_values(
        self,
        group_name: str,
        feature_name: str,
        values: Dict[str, Any],
    ) -> None:
        """
        Set feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            values: A dictionary mapping entity IDs to feature values.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Set the feature values
            for entity_id, value in values.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feature_values (group_name, feature_name, entity_id, value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        group_name,
                        feature_name,
                        entity_id,
                        json.dumps(value),
                    ),
                )
            
            # Commit the changes
            conn.commit()
    
    def delete_feature_values(
        self,
        group_name: str,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Delete feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            entity_ids: The entity IDs. If None, delete values for all entities.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the feature values
            if entity_ids:
                # Delete values for specific entities
                placeholders = ", ".join(["?"] * len(entity_ids))
                
                conn.execute(
                    f"""
                    DELETE FROM feature_values
                    WHERE group_name = ? AND feature_name = ? AND entity_id IN ({placeholders})
                    """,
                    (group_name, feature_name) + tuple(entity_ids),
                )
            else:
                # Delete values for all entities
                conn.execute(
                    """
                    DELETE FROM feature_values
                    WHERE group_name = ? AND feature_name = ?
                    """,
                    (group_name, feature_name),
                )
            
            # Commit the changes
            conn.commit()


class RedisFeatureStorage(FeatureStorageBackend):
    """
    Redis feature storage.
    
    This class implements a feature storage backend that stores features in Redis.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        """
        Initialize a Redis feature storage.
        
        Args:
            host: The Redis host.
            port: The Redis port.
            db: The Redis database.
            password: The Redis password.
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis = None
        self.lock = threading.RLock()
    
    def _get_redis(self) -> redis.Redis:
        """
        Get a connection to Redis.
        
        Returns:
            A connection to Redis.
        """
        if not self.redis:
            # Connect to Redis
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
            )
        
        return self.redis
    
    def _get_group_key(self, name: str) -> str:
        """
        Get the Redis key for a feature group.
        
        Args:
            name: The feature group name.
            
        Returns:
            The Redis key.
        """
        return f"feature_group:{name}"
    
    def _get_feature_key(self, group_name: str, feature_name: str, entity_id: str) -> str:
        """
        Get the Redis key for a feature value.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            entity_id: The entity ID.
            
        Returns:
            The Redis key.
        """
        return f"feature_value:{group_name}:{feature_name}:{entity_id}"
    
    def _get_feature_pattern(self, group_name: str, feature_name: str) -> str:
        """
        Get the Redis key pattern for feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            
        Returns:
            The Redis key pattern.
        """
        return f"feature_value:{group_name}:{feature_name}:*"
    
    def save_group(self, group: Any) -> None:
        """
        Save a feature group.
        
        Args:
            group: The feature group to save.
        """
        with self.lock:
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Convert the group to a dictionary
            group_dict = group.to_dict()
            
            # Save the group
            redis_conn.set(
                self._get_group_key(group.name),
                json.dumps(group_dict),
            )
    
    def get_group(self, name: str) -> Optional[Any]:
        """
        Get a feature group by name.
        
        Args:
            name: The feature group name.
            
        Returns:
            The feature group, or None if the group is not found.
        """
        with self.lock:
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Get the group
            group_data = redis_conn.get(self._get_group_key(name))
            
            if not group_data:
                return None
            
            # Parse the group data
            group_dict = json.loads(group_data)
            
            # Import the FeatureGroup class
            from feluda.feature_store.store import FeatureGroup
            
            # Create the group
            return FeatureGroup.from_dict(group_dict)
    
    def get_groups(self) -> Dict[str, Any]:
        """
        Get all feature groups.
        
        Returns:
            A dictionary mapping feature group names to feature groups.
        """
        with self.lock:
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Get all group keys
            group_keys = redis_conn.keys("feature_group:*")
            
            # Import the FeatureGroup class
            from feluda.feature_store.store import FeatureGroup
            
            # Create the groups
            groups = {}
            
            for key in group_keys:
                # Get the group name
                name = key.decode().split(":", 1)[1]
                
                # Get the group
                group = self.get_group(name)
                
                if group:
                    groups[name] = group
            
            return groups
    
    def delete_group(self, name: str) -> None:
        """
        Delete a feature group.
        
        Args:
            name: The feature group name.
        """
        with self.lock:
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Get the group
            group = self.get_group(name)
            
            if not group:
                return
            
            # Delete the group
            redis_conn.delete(self._get_group_key(name))
            
            # Delete the feature values
            for feature_name in group.features:
                # Get the feature value keys
                feature_keys = redis_conn.keys(self._get_feature_pattern(name, feature_name))
                
                if feature_keys:
                    # Delete the feature values
                    redis_conn.delete(*feature_keys)
    
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
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Get the feature values
            if entity_ids:
                # Get values for specific entities
                keys = [
                    self._get_feature_key(group_name, feature_name, entity_id)
                    for entity_id in entity_ids
                ]
                
                if not keys:
                    return {}
                
                values = redis_conn.mget(keys)
                
                # Create the result
                return {
                    entity_id: json.loads(value.decode()) if value else None
                    for entity_id, value in zip(entity_ids, values)
                    if value
                }
            else:
                # Get values for all entities
                keys = redis_conn.keys(self._get_feature_pattern(group_name, feature_name))
                
                if not keys:
                    return {}
                
                values = redis_conn.mget(keys)
                
                # Create the result
                return {
                    key.decode().split(":")[-1]: json.loads(value.decode()) if value else None
                    for key, value in zip(keys, values)
                    if value
                }
    
    def set_feature_values(
        self,
        group_name: str,
        feature_name: str,
        values: Dict[str, Any],
    ) -> None:
        """
        Set feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            values: A dictionary mapping entity IDs to feature values.
        """
        with self.lock:
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Set the feature values
            pipeline = redis_conn.pipeline()
            
            for entity_id, value in values.items():
                pipeline.set(
                    self._get_feature_key(group_name, feature_name, entity_id),
                    json.dumps(value),
                )
            
            pipeline.execute()
    
    def delete_feature_values(
        self,
        group_name: str,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Delete feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            entity_ids: The entity IDs. If None, delete values for all entities.
        """
        with self.lock:
            # Get a connection to Redis
            redis_conn = self._get_redis()
            
            # Delete the feature values
            if entity_ids:
                # Delete values for specific entities
                keys = [
                    self._get_feature_key(group_name, feature_name, entity_id)
                    for entity_id in entity_ids
                ]
                
                if keys:
                    redis_conn.delete(*keys)
            else:
                # Delete values for all entities
                keys = redis_conn.keys(self._get_feature_pattern(group_name, feature_name))
                
                if keys:
                    redis_conn.delete(*keys)


class FeatureStorage:
    """
    Feature storage.
    
    This class is responsible for storing and loading features.
    """
    
    def __init__(self, backend: Optional[FeatureStorageBackend] = None):
        """
        Initialize the feature storage.
        
        Args:
            backend: The feature storage backend.
        """
        config = get_config()
        
        if backend:
            self.backend = backend
        elif config.feature_storage_type == "redis":
            self.backend = RedisFeatureStorage(
                host=config.feature_storage_host or "localhost",
                port=int(config.feature_storage_port or 6379),
                db=int(config.feature_storage_db or 0),
                password=config.feature_storage_password,
            )
        else:
            self.backend = SQLiteFeatureStorage(
                db_path=config.feature_storage_db or "features/features.db",
            )
    
    def save_group(self, group: Any) -> None:
        """
        Save a feature group.
        
        Args:
            group: The feature group to save.
        """
        self.backend.save_group(group)
    
    def get_group(self, name: str) -> Optional[Any]:
        """
        Get a feature group by name.
        
        Args:
            name: The feature group name.
            
        Returns:
            The feature group, or None if the group is not found.
        """
        return self.backend.get_group(name)
    
    def get_groups(self) -> Dict[str, Any]:
        """
        Get all feature groups.
        
        Returns:
            A dictionary mapping feature group names to feature groups.
        """
        return self.backend.get_groups()
    
    def delete_group(self, name: str) -> None:
        """
        Delete a feature group.
        
        Args:
            name: The feature group name.
        """
        self.backend.delete_group(name)
    
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
        return self.backend.get_feature_values(group_name, feature_name, entity_ids)
    
    def set_feature_values(
        self,
        group_name: str,
        feature_name: str,
        values: Dict[str, Any],
    ) -> None:
        """
        Set feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            values: A dictionary mapping entity IDs to feature values.
        """
        self.backend.set_feature_values(group_name, feature_name, values)
    
    def delete_feature_values(
        self,
        group_name: str,
        feature_name: str,
        entity_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Delete feature values.
        
        Args:
            group_name: The feature group name.
            feature_name: The feature name.
            entity_ids: The entity IDs. If None, delete values for all entities.
        """
        self.backend.delete_feature_values(group_name, feature_name, entity_ids)


# Global feature storage instance
_feature_storage = None
_feature_storage_lock = threading.RLock()


def get_feature_storage() -> FeatureStorage:
    """
    Get the global feature storage instance.
    
    Returns:
        The global feature storage instance.
    """
    global _feature_storage
    
    with _feature_storage_lock:
        if _feature_storage is None:
            _feature_storage = FeatureStorage()
        
        return _feature_storage
