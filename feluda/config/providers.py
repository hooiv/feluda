"""
Configuration providers for Feluda.

This module provides providers for configuration values.
"""

import abc
import enum
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field

from feluda.observability import get_logger

log = get_logger(__name__)


class ConfigProvider(abc.ABC):
    """
    Base class for configuration providers.
    
    This class defines the interface for configuration providers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key.
            default: The default value.
            
        Returns:
            The configuration value, or the default value if the key is not found.
        """
        pass
    
    @abc.abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key.
            value: The configuration value.
        """
        pass
    
    @abc.abstractmethod
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The configuration key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete a configuration key.
        
        Args:
            key: The configuration key.
        """
        pass
    
    @abc.abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            A dictionary mapping configuration keys to values.
        """
        pass


class FileConfigProvider(ConfigProvider):
    """
    File configuration provider.
    
    This class implements a configuration provider that loads configuration from a file.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize a file configuration provider.
        
        Args:
            file_path: The file path.
        """
        self.file_path = file_path
        self.config: Dict[str, Any] = {}
        self.lock = threading.RLock()
        
        # Load the configuration
        self._load()
    
    def _load(self) -> None:
        """
        Load the configuration from the file.
        """
        with self.lock:
            if not os.path.isfile(self.file_path):
                return
            
            try:
                with open(self.file_path, "r") as f:
                    if self.file_path.endswith(".json"):
                        self.config = json.load(f)
                    elif self.file_path.endswith((".yaml", ".yml")):
                        self.config = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {self.file_path}")
            
            except Exception as e:
                log.error(f"Failed to load configuration from {self.file_path}: {e}")
    
    def _save(self) -> None:
        """
        Save the configuration to the file.
        """
        with self.lock:
            try:
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                
                with open(self.file_path, "w") as f:
                    if self.file_path.endswith(".json"):
                        json.dump(self.config, f, indent=2)
                    elif self.file_path.endswith((".yaml", ".yml")):
                        yaml.dump(self.config, f, default_flow_style=False)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {self.file_path}")
            
            except Exception as e:
                log.error(f"Failed to save configuration to {self.file_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key.
            default: The default value.
            
        Returns:
            The configuration value, or the default value if the key is not found.
        """
        with self.lock:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key.
            value: The configuration value.
        """
        with self.lock:
            self.config[key] = value
            self._save()
    
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The configuration key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        with self.lock:
            return key in self.config
    
    def delete(self, key: str) -> None:
        """
        Delete a configuration key.
        
        Args:
            key: The configuration key.
        """
        with self.lock:
            if key in self.config:
                del self.config[key]
                self._save()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            A dictionary mapping configuration keys to values.
        """
        with self.lock:
            return self.config.copy()


class EnvConfigProvider(ConfigProvider):
    """
    Environment configuration provider.
    
    This class implements a configuration provider that loads configuration from environment variables.
    """
    
    def __init__(self, prefix: str = "FELUDA_"):
        """
        Initialize an environment configuration provider.
        
        Args:
            prefix: The environment variable prefix.
        """
        self.prefix = prefix
        self.lock = threading.RLock()
    
    def _get_env_key(self, key: str) -> str:
        """
        Get the environment variable key.
        
        Args:
            key: The configuration key.
            
        Returns:
            The environment variable key.
        """
        return f"{self.prefix}{key.upper()}"
    
    def _get_config_key(self, env_key: str) -> str:
        """
        Get the configuration key.
        
        Args:
            env_key: The environment variable key.
            
        Returns:
            The configuration key.
        """
        return env_key[len(self.prefix):].lower()
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert a string value to a Python value.
        
        Args:
            value: The string value.
            
        Returns:
            The Python value.
        """
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False
        elif value.isdigit():
            return int(value)
        elif value.replace(".", "", 1).isdigit():
            return float(value)
        else:
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key.
            default: The default value.
            
        Returns:
            The configuration value, or the default value if the key is not found.
        """
        with self.lock:
            env_key = self._get_env_key(key)
            
            if env_key in os.environ:
                return self._convert_value(os.environ[env_key])
            else:
                return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key.
            value: The configuration value.
        """
        with self.lock:
            env_key = self._get_env_key(key)
            os.environ[env_key] = str(value)
    
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The configuration key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        with self.lock:
            env_key = self._get_env_key(key)
            return env_key in os.environ
    
    def delete(self, key: str) -> None:
        """
        Delete a configuration key.
        
        Args:
            key: The configuration key.
        """
        with self.lock:
            env_key = self._get_env_key(key)
            
            if env_key in os.environ:
                del os.environ[env_key]
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            A dictionary mapping configuration keys to values.
        """
        with self.lock:
            result = {}
            
            for env_key, value in os.environ.items():
                if env_key.startswith(self.prefix):
                    config_key = self._get_config_key(env_key)
                    result[config_key] = self._convert_value(value)
            
            return result


class DictConfigProvider(ConfigProvider):
    """
    Dictionary configuration provider.
    
    This class implements a configuration provider that loads configuration from a dictionary.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a dictionary configuration provider.
        
        Args:
            config: The configuration dictionary.
        """
        self.config = config or {}
        self.lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key.
            default: The default value.
            
        Returns:
            The configuration value, or the default value if the key is not found.
        """
        with self.lock:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key.
            value: The configuration value.
        """
        with self.lock:
            self.config[key] = value
    
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The configuration key.
            
        Returns:
            True if the key exists, False otherwise.
        """
        with self.lock:
            return key in self.config
    
    def delete(self, key: str) -> None:
        """
        Delete a configuration key.
        
        Args:
            key: The configuration key.
        """
        with self.lock:
            if key in self.config:
                del self.config[key]
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            A dictionary mapping configuration keys to values.
        """
        with self.lock:
            return self.config.copy()


# Global configuration provider instance
_config_provider = None
_config_provider_lock = threading.RLock()


def get_config_provider() -> ConfigProvider:
    """
    Get the global configuration provider instance.
    
    Returns:
        The global configuration provider instance.
    """
    global _config_provider
    
    with _config_provider_lock:
        if _config_provider is None:
            # Try to load configuration from a file
            config_file = os.environ.get("FELUDA_CONFIG_FILE")
            
            if config_file and os.path.isfile(config_file):
                _config_provider = FileConfigProvider(config_file)
            else:
                # Use environment variables
                _config_provider = EnvConfigProvider()
        
        return _config_provider
