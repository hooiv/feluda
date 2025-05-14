"""
Plugin base class for Feluda.

This module provides the base class for Feluda plugins.
"""

import abc
import enum
import importlib
import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.observability import get_logger

log = get_logger(__name__)


class PluginType(str, enum.Enum):
    """Enum for plugin types."""
    
    OPERATOR = "operator"
    VERIFIER = "verifier"
    RESILIENCE = "resilience"
    OBSERVABILITY = "observability"
    PERFORMANCE = "performance"
    CRYPTO = "crypto"
    AI_AGENT = "ai_agent"
    TESTING = "testing"
    HARDWARE = "hardware"
    AUTONOMIC = "autonomic"
    CLI = "cli"
    API = "api"
    DASHBOARD = "dashboard"
    OTHER = "other"


class PluginInfo(BaseModel):
    """Information about a plugin."""
    
    name: str = Field(..., description="The name of the plugin")
    version: str = Field(..., description="The version of the plugin")
    description: str = Field(..., description="A description of the plugin")
    author: str = Field(..., description="The author of the plugin")
    plugin_type: PluginType = Field(..., description="The type of the plugin")
    entry_point: str = Field(..., description="The entry point of the plugin")
    dependencies: List[str] = Field(default_factory=list, description="The dependencies of the plugin")
    homepage: Optional[str] = Field(None, description="The homepage of the plugin")
    repository: Optional[str] = Field(None, description="The repository of the plugin")
    license: Optional[str] = Field(None, description="The license of the plugin")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the plugin info to a dictionary.
        
        Returns:
            A dictionary representation of the plugin info.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginInfo":
        """
        Create a plugin info from a dictionary.
        
        Args:
            data: The dictionary to create the plugin info from.
            
        Returns:
            A plugin info.
        """
        return cls(**data)


class PluginConfig(BaseModel):
    """Configuration for a plugin."""
    
    enabled: bool = Field(True, description="Whether the plugin is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="The configuration for the plugin")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the plugin config to a dictionary.
        
        Returns:
            A dictionary representation of the plugin config.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginConfig":
        """
        Create a plugin config from a dictionary.
        
        Args:
            data: The dictionary to create the plugin config from.
            
        Returns:
            A plugin config.
        """
        return cls(**data)


class FeludaPlugin(abc.ABC):
    """
    Base class for Feluda plugins.
    
    This class defines the interface for Feluda plugins.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize a plugin.
        
        Args:
            config: The configuration for the plugin.
        """
        self.config = config or PluginConfig()
    
    @property
    @abc.abstractmethod
    def info(self) -> PluginInfo:
        """
        Get information about the plugin.
        
        Returns:
            Information about the plugin.
        """
        pass
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize the plugin.
        
        This method is called when the plugin is loaded.
        """
        pass
    
    @abc.abstractmethod
    def shutdown(self) -> None:
        """
        Shut down the plugin.
        
        This method is called when the plugin is unloaded.
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if the plugin is enabled.
        
        Returns:
            True if the plugin is enabled, False otherwise.
        """
        return self.config.enabled
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the plugin configuration.
        
        Returns:
            The plugin configuration.
        """
        return self.config.config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the plugin configuration.
        
        Args:
            config: The plugin configuration.
        """
        self.config.config = config


def load_plugin_class(entry_point: str) -> Type[FeludaPlugin]:
    """
    Load a plugin class from an entry point.
    
    Args:
        entry_point: The entry point of the plugin.
        
    Returns:
        The plugin class.
        
    Raises:
        ImportError: If the plugin class cannot be imported.
        TypeError: If the plugin class is not a subclass of FeludaPlugin.
    """
    try:
        module_name, class_name = entry_point.rsplit(".", 1)
        module = importlib.import_module(module_name)
        plugin_class = getattr(module, class_name)
        
        if not inspect.isclass(plugin_class) or not issubclass(plugin_class, FeludaPlugin):
            raise TypeError(f"Plugin class {entry_point} is not a subclass of FeludaPlugin")
        
        return plugin_class
    
    except (ImportError, AttributeError) as e:
        log.error(f"Failed to load plugin class {entry_point}: {e}")
        raise ImportError(f"Failed to load plugin class {entry_point}: {e}")


def create_plugin(plugin_class: Type[FeludaPlugin], config: Optional[PluginConfig] = None) -> FeludaPlugin:
    """
    Create a plugin instance.
    
    Args:
        plugin_class: The plugin class.
        config: The plugin configuration.
        
    Returns:
        A plugin instance.
        
    Raises:
        Exception: If the plugin instance cannot be created.
    """
    try:
        plugin = plugin_class(config)
        return plugin
    
    except Exception as e:
        log.error(f"Failed to create plugin instance of {plugin_class.__name__}: {e}")
        raise Exception(f"Failed to create plugin instance of {plugin_class.__name__}: {e}")


def load_plugin(entry_point: str, config: Optional[PluginConfig] = None) -> FeludaPlugin:
    """
    Load a plugin from an entry point.
    
    Args:
        entry_point: The entry point of the plugin.
        config: The plugin configuration.
        
    Returns:
        A plugin instance.
        
    Raises:
        ImportError: If the plugin class cannot be imported.
        Exception: If the plugin instance cannot be created.
    """
    plugin_class = load_plugin_class(entry_point)
    return create_plugin(plugin_class, config)
