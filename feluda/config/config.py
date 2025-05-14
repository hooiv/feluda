"""
Configuration management for Feluda.

This module provides configuration management for Feluda.
"""

import enum
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field

from feluda.observability import get_logger

log = get_logger(__name__)


class ConfigSource(str, enum.Enum):
    """Enum for configuration sources."""
    
    FILE = "file"
    ENVIRONMENT = "environment"
    COMMAND_LINE = "command_line"
    DEFAULT = "default"


class FeludaConfig(BaseModel):
    """
    Configuration for Feluda.
    
    This class defines the configuration for Feluda.
    """
    
    # General configuration
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Log level")
    log_file: Optional[str] = Field(None, description="Log file")
    
    # API configuration
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")
    api_workers: int = Field(4, description="API workers")
    api_timeout: int = Field(60, description="API timeout in seconds")
    
    # Dashboard configuration
    dashboard_host: str = Field("0.0.0.0", description="Dashboard host")
    dashboard_port: int = Field(8050, description="Dashboard port")
    
    # Database configuration
    database_url: Optional[str] = Field(None, description="Database URL")
    database_pool_size: int = Field(5, description="Database connection pool size")
    database_max_overflow: int = Field(10, description="Database connection max overflow")
    
    # Cache configuration
    cache_url: Optional[str] = Field(None, description="Cache URL")
    cache_ttl: int = Field(300, description="Cache TTL in seconds")
    
    # Queue configuration
    queue_url: Optional[str] = Field(None, description="Queue URL")
    queue_name: str = Field("feluda", description="Queue name")
    
    # Storage configuration
    storage_url: Optional[str] = Field(None, description="Storage URL")
    storage_bucket: str = Field("feluda", description="Storage bucket")
    
    # Security configuration
    secret_key: Optional[str] = Field(None, description="Secret key")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration: int = Field(3600, description="JWT expiration in seconds")
    
    # Observability configuration
    telemetry_enabled: bool = Field(False, description="Enable telemetry")
    telemetry_url: Optional[str] = Field(None, description="Telemetry URL")
    metrics_enabled: bool = Field(False, description="Enable metrics")
    metrics_url: Optional[str] = Field(None, description="Metrics URL")
    tracing_enabled: bool = Field(False, description="Enable tracing")
    tracing_url: Optional[str] = Field(None, description="Tracing URL")
    
    # Plugin configuration
    plugin_dirs: List[str] = Field(default_factory=list, description="Plugin directories")
    plugin_config_file: Optional[str] = Field(None, description="Plugin configuration file")
    
    # Operator configuration
    operator_config_file: Optional[str] = Field(None, description="Operator configuration file")
    
    # Custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            A dictionary representation of the configuration.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeludaConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            data: The dictionary to create the configuration from.
            
        Returns:
            A configuration.
        """
        return cls(**data)
    
    def to_json(self) -> str:
        """
        Convert the configuration to JSON.
        
        Returns:
            A JSON representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "FeludaConfig":
        """
        Create a configuration from JSON.
        
        Args:
            data: The JSON to create the configuration from.
            
        Returns:
            A configuration.
        """
        return cls.from_dict(json.loads(data))
    
    def to_yaml(self) -> str:
        """
        Convert the configuration to YAML.
        
        Returns:
            A YAML representation of the configuration.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, data: str) -> "FeludaConfig":
        """
        Create a configuration from YAML.
        
        Args:
            data: The YAML to create the configuration from.
            
        Returns:
            A configuration.
        """
        return cls.from_dict(yaml.safe_load(data))
    
    def update(self, other: Union[Dict[str, Any], "FeludaConfig"]) -> "FeludaConfig":
        """
        Update the configuration with another configuration.
        
        Args:
            other: The configuration to update with.
            
        Returns:
            The updated configuration.
        """
        if isinstance(other, FeludaConfig):
            other = other.to_dict()
        
        data = self.to_dict()
        data.update(other)
        
        return FeludaConfig.from_dict(data)


class ConfigManager:
    """
    Configuration manager for Feluda.
    
    This class is responsible for loading and managing Feluda configuration.
    """
    
    def __init__(self):
        """
        Initialize a configuration manager.
        """
        self.config = FeludaConfig()
        self.sources: Dict[str, ConfigSource] = {}
        self.lock = threading.RLock()
    
    def load_from_file(self, file_path: str) -> FeludaConfig:
        """
        Load configuration from a file.
        
        Args:
            file_path: The file path.
            
        Returns:
            The loaded configuration.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        with self.lock:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Configuration file {file_path} does not exist")
            
            try:
                with open(file_path, "r") as f:
                    if file_path.endswith(".json"):
                        config_data = json.load(f)
                    elif file_path.endswith((".yaml", ".yml")):
                        config_data = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {file_path}")
                
                # Create a configuration
                config = FeludaConfig.from_dict(config_data)
                
                # Update the configuration
                self.config = self.config.update(config)
                
                # Update the sources
                for key in config_data.keys():
                    self.sources[key] = ConfigSource.FILE
                
                return self.config
            
            except Exception as e:
                log.error(f"Failed to load configuration from {file_path}: {e}")
                raise
    
    def load_from_env(self, prefix: str = "FELUDA_") -> FeludaConfig:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: The environment variable prefix.
            
        Returns:
            The loaded configuration.
        """
        with self.lock:
            try:
                # Get environment variables
                env_vars = {}
                
                for key, value in os.environ.items():
                    if key.startswith(prefix):
                        # Remove the prefix
                        config_key = key[len(prefix):].lower()
                        
                        # Convert the value
                        if value.lower() in ("true", "yes", "1"):
                            env_vars[config_key] = True
                        elif value.lower() in ("false", "no", "0"):
                            env_vars[config_key] = False
                        elif value.isdigit():
                            env_vars[config_key] = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            env_vars[config_key] = float(value)
                        else:
                            env_vars[config_key] = value
                
                # Update the configuration
                self.config = self.config.update(env_vars)
                
                # Update the sources
                for key in env_vars.keys():
                    self.sources[key] = ConfigSource.ENVIRONMENT
                
                return self.config
            
            except Exception as e:
                log.error(f"Failed to load configuration from environment variables: {e}")
                raise
    
    def load_from_args(self, args: Dict[str, Any]) -> FeludaConfig:
        """
        Load configuration from command-line arguments.
        
        Args:
            args: The command-line arguments.
            
        Returns:
            The loaded configuration.
        """
        with self.lock:
            try:
                # Update the configuration
                self.config = self.config.update(args)
                
                # Update the sources
                for key in args.keys():
                    self.sources[key] = ConfigSource.COMMAND_LINE
                
                return self.config
            
            except Exception as e:
                log.error(f"Failed to load configuration from command-line arguments: {e}")
                raise
    
    def get_config(self) -> FeludaConfig:
        """
        Get the configuration.
        
        Returns:
            The configuration.
        """
        with self.lock:
            return self.config
    
    def get_source(self, key: str) -> Optional[ConfigSource]:
        """
        Get the source of a configuration key.
        
        Args:
            key: The configuration key.
            
        Returns:
            The source of the configuration key, or None if the key is not found.
        """
        with self.lock:
            return self.sources.get(key)
    
    def get_sources(self) -> Dict[str, ConfigSource]:
        """
        Get the sources of all configuration keys.
        
        Returns:
            A dictionary mapping configuration keys to sources.
        """
        with self.lock:
            return self.sources.copy()


# Global configuration manager instance
_config_manager = None
_config_manager_lock = threading.RLock()


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        The global configuration manager instance.
    """
    global _config_manager
    
    with _config_manager_lock:
        if _config_manager is None:
            _config_manager = ConfigManager()
        
        return _config_manager


def get_config() -> FeludaConfig:
    """
    Get the global configuration.
    
    Returns:
        The global configuration.
    """
    return get_config_manager().get_config()


def load_config(
    file_path: Optional[str] = None,
    env_prefix: str = "FELUDA_",
    args: Optional[Dict[str, Any]] = None,
) -> FeludaConfig:
    """
    Load configuration from various sources.
    
    Args:
        file_path: The configuration file path.
        env_prefix: The environment variable prefix.
        args: The command-line arguments.
        
    Returns:
        The loaded configuration.
    """
    config_manager = get_config_manager()
    
    # Load configuration from file
    if file_path:
        config_manager.load_from_file(file_path)
    
    # Load configuration from environment variables
    config_manager.load_from_env(env_prefix)
    
    # Load configuration from command-line arguments
    if args:
        config_manager.load_from_args(args)
    
    return config_manager.get_config()
