"""
Plugin manager for Feluda.

This module provides a plugin manager for loading and managing Feluda plugins.
"""

import importlib
import inspect
import json
import logging
import os
import pkgutil
import sys
import threading
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

import pkg_resources
import yaml

from feluda.observability import get_logger
from feluda.plugins.plugin import (
    FeludaPlugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    create_plugin,
    load_plugin,
    load_plugin_class,
)

log = get_logger(__name__)


class PluginManager:
    """
    Plugin manager for Feluda.
    
    This class is responsible for loading and managing Feluda plugins.
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize a plugin manager.
        
        Args:
            plugin_dirs: Directories to search for plugins.
        """
        self.plugin_dirs = plugin_dirs or []
        self.plugins: Dict[str, FeludaPlugin] = {}
        self.plugin_classes: Dict[str, Type[FeludaPlugin]] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.lock = threading.RLock()
    
    def add_plugin_dir(self, plugin_dir: str) -> None:
        """
        Add a directory to search for plugins.
        
        Args:
            plugin_dir: The directory to add.
        """
        with self.lock:
            if plugin_dir not in self.plugin_dirs:
                self.plugin_dirs.append(plugin_dir)
    
    def discover_plugins(self) -> Dict[str, Type[FeludaPlugin]]:
        """
        Discover plugins in the plugin directories.
        
        Returns:
            A dictionary mapping plugin names to plugin classes.
        """
        with self.lock:
            # Discover plugins in the plugin directories
            for plugin_dir in self.plugin_dirs:
                if not os.path.isdir(plugin_dir):
                    log.warning(f"Plugin directory {plugin_dir} does not exist")
                    continue
                
                # Add the plugin directory to the Python path
                if plugin_dir not in sys.path:
                    sys.path.insert(0, plugin_dir)
                
                # Discover plugins in the plugin directory
                for _, name, is_pkg in pkgutil.iter_modules([plugin_dir]):
                    if not is_pkg:
                        continue
                    
                    try:
                        # Import the plugin module
                        module = importlib.import_module(name)
                        
                        # Find plugin classes in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            
                            if (
                                inspect.isclass(attr)
                                and issubclass(attr, FeludaPlugin)
                                and attr is not FeludaPlugin
                            ):
                                # Create a plugin instance
                                plugin_class = attr
                                plugin_name = f"{name}.{attr_name}"
                                
                                # Add the plugin class to the dictionary
                                self.plugin_classes[plugin_name] = plugin_class
                    
                    except Exception as e:
                        log.error(f"Failed to discover plugins in module {name}: {e}")
            
            # Discover plugins using entry points
            for entry_point in pkg_resources.iter_entry_points("feluda.plugins"):
                try:
                    # Load the plugin class
                    plugin_class = entry_point.load()
                    
                    # Add the plugin class to the dictionary
                    self.plugin_classes[entry_point.name] = plugin_class
                
                except Exception as e:
                    log.error(f"Failed to discover plugin {entry_point.name}: {e}")
            
            return self.plugin_classes
    
    def load_plugin_configs(self, config_file: str) -> Dict[str, PluginConfig]:
        """
        Load plugin configurations from a file.
        
        Args:
            config_file: The configuration file.
            
        Returns:
            A dictionary mapping plugin names to plugin configurations.
        """
        with self.lock:
            try:
                # Load the configuration file
                with open(config_file, "r") as f:
                    if config_file.endswith(".json"):
                        config_data = json.load(f)
                    elif config_file.endswith((".yaml", ".yml")):
                        config_data = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {config_file}")
                
                # Parse the plugin configurations
                plugin_configs = {}
                
                for plugin_name, plugin_config in config_data.get("plugins", {}).items():
                    plugin_configs[plugin_name] = PluginConfig.from_dict(plugin_config)
                
                # Update the plugin configurations
                self.plugin_configs.update(plugin_configs)
                
                return plugin_configs
            
            except Exception as e:
                log.error(f"Failed to load plugin configurations from {config_file}: {e}")
                return {}
    
    def load_plugins(self) -> Dict[str, FeludaPlugin]:
        """
        Load plugins.
        
        Returns:
            A dictionary mapping plugin names to plugin instances.
        """
        with self.lock:
            # Discover plugins
            self.discover_plugins()
            
            # Load plugins
            for plugin_name, plugin_class in self.plugin_classes.items():
                try:
                    # Get the plugin configuration
                    plugin_config = self.plugin_configs.get(plugin_name)
                    
                    # Skip disabled plugins
                    if plugin_config and not plugin_config.enabled:
                        log.info(f"Skipping disabled plugin {plugin_name}")
                        continue
                    
                    # Create a plugin instance
                    plugin = create_plugin(plugin_class, plugin_config)
                    
                    # Initialize the plugin
                    plugin.initialize()
                    
                    # Add the plugin to the dictionary
                    self.plugins[plugin_name] = plugin
                    
                    log.info(f"Loaded plugin {plugin_name} v{plugin.info.version}")
                
                except Exception as e:
                    log.error(f"Failed to load plugin {plugin_name}: {e}")
            
            return self.plugins
    
    def get_plugin(self, plugin_name: str) -> Optional[FeludaPlugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: The name of the plugin.
            
        Returns:
            The plugin, or None if the plugin is not found.
        """
        with self.lock:
            return self.plugins.get(plugin_name)
    
    def get_plugins(self, plugin_type: Optional[PluginType] = None) -> Dict[str, FeludaPlugin]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: The type of plugins to get. If None, get all plugins.
            
        Returns:
            A dictionary mapping plugin names to plugin instances.
        """
        with self.lock:
            if plugin_type is None:
                return self.plugins.copy()
            
            return {
                name: plugin
                for name, plugin in self.plugins.items()
                if plugin.info.plugin_type == plugin_type
            }
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: The name of the plugin.
            
        Returns:
            True if the plugin was unloaded, False otherwise.
        """
        with self.lock:
            plugin = self.plugins.get(plugin_name)
            
            if plugin is None:
                return False
            
            try:
                # Shut down the plugin
                plugin.shutdown()
                
                # Remove the plugin from the dictionary
                del self.plugins[plugin_name]
                
                log.info(f"Unloaded plugin {plugin_name}")
                
                return True
            
            except Exception as e:
                log.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
    
    def unload_plugins(self) -> None:
        """
        Unload all plugins.
        """
        with self.lock:
            for plugin_name in list(self.plugins.keys()):
                self.unload_plugin(plugin_name)


# Global plugin manager instance
_plugin_manager = None
_plugin_manager_lock = threading.RLock()


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.
    
    Returns:
        The global plugin manager instance.
    """
    global _plugin_manager
    
    with _plugin_manager_lock:
        if _plugin_manager is None:
            _plugin_manager = PluginManager()
        
        return _plugin_manager
