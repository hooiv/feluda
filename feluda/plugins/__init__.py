"""
Plugin system for Feluda.

This module provides a plugin system for extending Feluda's functionality.
"""

from feluda.plugins.manager import PluginManager, get_plugin_manager
from feluda.plugins.plugin import FeludaPlugin, PluginConfig, PluginInfo, PluginType

__all__ = [
    "FeludaPlugin",
    "PluginConfig",
    "PluginInfo",
    "PluginManager",
    "PluginType",
    "get_plugin_manager",
]
