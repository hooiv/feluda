"""
Workflow storage for Feluda.

This module provides storage for the workflow engine.
"""

import abc
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from feluda.observability import get_logger

log = get_logger(__name__)


class StorageBackend(abc.ABC):
    """
    Base class for storage backends.
    
    This class defines the interface for storage backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def save(self, key: str, value: Any) -> None:
        """
        Save a value.
        
        Args:
            key: The key.
            value: The value.
        """
        pass
    
    @abc.abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """
        Load a value.
        
        Args:
            key: The key.
            
        Returns:
            The value, or None if the key is not found.
        """
        pass
    
    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete a value.
        
        Args:
            key: The key.
        """
        pass
    
    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: The key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys with a prefix.
        
        Args:
            prefix: The key prefix.
            
        Returns:
            A list of keys.
        """
        pass


class FileStorageBackend(StorageBackend):
    """
    File storage backend.
    
    This class implements a storage backend that stores data in files.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize a file storage backend.
        
        Args:
            base_dir: The base directory.
        """
        self.base_dir = base_dir
        
        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """
        Get the file path for a key.
        
        Args:
            key: The key.
            
        Returns:
            The file path.
        """
        # Replace slashes with underscores to avoid directory traversal
        safe_key = key.replace("/", "_").replace("\\", "_")
        
        return os.path.join(self.base_dir, safe_key)
    
    def save(self, key: str, value: Any) -> None:
        """
        Save a value.
        
        Args:
            key: The key.
            value: The value.
        """
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, "w") as f:
                json.dump(value, f)
        
        except Exception as e:
            log.error(f"Error saving value for key {key}: {e}")
            raise
    
    def load(self, key: str) -> Optional[Any]:
        """
        Load a value.
        
        Args:
            key: The key.
            
        Returns:
            The value, or None if the key is not found.
        """
        file_path = self._get_file_path(key)
        
        if not os.path.isfile(file_path):
            return None
        
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        
        except Exception as e:
            log.error(f"Error loading value for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> None:
        """
        Delete a value.
        
        Args:
            key: The key.
        """
        file_path = self._get_file_path(key)
        
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            
            except Exception as e:
                log.error(f"Error deleting value for key {key}: {e}")
                raise
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: The key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        file_path = self._get_file_path(key)
        return os.path.isfile(file_path)
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys with a prefix.
        
        Args:
            prefix: The key prefix.
            
        Returns:
            A list of keys.
        """
        keys = []
        
        for file_name in os.listdir(self.base_dir):
            if file_name.startswith(prefix):
                keys.append(file_name)
        
        return keys


class MemoryStorageBackend(StorageBackend):
    """
    Memory storage backend.
    
    This class implements a storage backend that stores data in memory.
    """
    
    def __init__(self):
        """
        Initialize a memory storage backend.
        """
        self.storage: Dict[str, Any] = {}
        self.lock = threading.RLock()
    
    def save(self, key: str, value: Any) -> None:
        """
        Save a value.
        
        Args:
            key: The key.
            value: The value.
        """
        with self.lock:
            self.storage[key] = value
    
    def load(self, key: str) -> Optional[Any]:
        """
        Load a value.
        
        Args:
            key: The key.
            
        Returns:
            The value, or None if the key is not found.
        """
        with self.lock:
            return self.storage.get(key)
    
    def delete(self, key: str) -> None:
        """
        Delete a value.
        
        Args:
            key: The key.
        """
        with self.lock:
            if key in self.storage:
                del self.storage[key]
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: The key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        with self.lock:
            return key in self.storage
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys with a prefix.
        
        Args:
            prefix: The key prefix.
            
        Returns:
            A list of keys.
        """
        with self.lock:
            return [key for key in self.storage.keys() if key.startswith(prefix)]


class StorageManager:
    """
    Storage manager.
    
    This class is responsible for managing storage backends.
    """
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        """
        Initialize the storage manager.
        
        Args:
            backend: The storage backend.
        """
        self.backend = backend or MemoryStorageBackend()
        self.lock = threading.RLock()
    
    def save(self, key: str, value: Any) -> None:
        """
        Save a value.
        
        Args:
            key: The key.
            value: The value.
        """
        with self.lock:
            self.backend.save(key, value)
    
    def load(self, key: str) -> Optional[Any]:
        """
        Load a value.
        
        Args:
            key: The key.
            
        Returns:
            The value, or None if the key is not found.
        """
        with self.lock:
            return self.backend.load(key)
    
    def delete(self, key: str) -> None:
        """
        Delete a value.
        
        Args:
            key: The key.
        """
        with self.lock:
            self.backend.delete(key)
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: The key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        with self.lock:
            return self.backend.exists(key)
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys with a prefix.
        
        Args:
            prefix: The key prefix.
            
        Returns:
            A list of keys.
        """
        with self.lock:
            return self.backend.list_keys(prefix)
    
    def save_workflow_definition(self, workflow_id: str, definition: Dict[str, Any]) -> None:
        """
        Save a workflow definition.
        
        Args:
            workflow_id: The workflow ID.
            definition: The workflow definition.
        """
        key = f"workflow:{workflow_id}:definition"
        self.save(key, definition)
    
    def load_workflow_definition(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow definition.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            The workflow definition, or None if the workflow is not found.
        """
        key = f"workflow:{workflow_id}:definition"
        return self.load(key)
    
    def save_workflow_execution(self, workflow_id: str, execution_id: str, execution: Dict[str, Any]) -> None:
        """
        Save a workflow execution.
        
        Args:
            workflow_id: The workflow ID.
            execution_id: The execution ID.
            execution: The workflow execution.
        """
        key = f"workflow:{workflow_id}:execution:{execution_id}"
        self.save(key, execution)
    
    def load_workflow_execution(self, workflow_id: str, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow execution.
        
        Args:
            workflow_id: The workflow ID.
            execution_id: The execution ID.
            
        Returns:
            The workflow execution, or None if the execution is not found.
        """
        key = f"workflow:{workflow_id}:execution:{execution_id}"
        return self.load(key)
    
    def list_workflow_executions(self, workflow_id: str) -> List[str]:
        """
        List workflow executions.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            A list of execution IDs.
        """
        prefix = f"workflow:{workflow_id}:execution:"
        keys = self.list_keys(prefix)
        return [key[len(prefix):] for key in keys]
    
    def save_task_execution(self, workflow_id: str, execution_id: str, task_id: str, execution: Dict[str, Any]) -> None:
        """
        Save a task execution.
        
        Args:
            workflow_id: The workflow ID.
            execution_id: The execution ID.
            task_id: The task ID.
            execution: The task execution.
        """
        key = f"workflow:{workflow_id}:execution:{execution_id}:task:{task_id}"
        self.save(key, execution)
    
    def load_task_execution(self, workflow_id: str, execution_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a task execution.
        
        Args:
            workflow_id: The workflow ID.
            execution_id: The execution ID.
            task_id: The task ID.
            
        Returns:
            The task execution, or None if the execution is not found.
        """
        key = f"workflow:{workflow_id}:execution:{execution_id}:task:{task_id}"
        return self.load(key)
    
    def list_task_executions(self, workflow_id: str, execution_id: str) -> List[str]:
        """
        List task executions.
        
        Args:
            workflow_id: The workflow ID.
            execution_id: The execution ID.
            
        Returns:
            A list of task IDs.
        """
        prefix = f"workflow:{workflow_id}:execution:{execution_id}:task:"
        keys = self.list_keys(prefix)
        return [key[len(prefix):] for key in keys]


# Global storage manager instance
_storage_manager = None
_storage_manager_lock = threading.RLock()


def get_storage_manager() -> StorageManager:
    """
    Get the global storage manager instance.
    
    Returns:
        The global storage manager instance.
    """
    global _storage_manager
    
    with _storage_manager_lock:
        if _storage_manager is None:
            _storage_manager = StorageManager()
        
        return _storage_manager
