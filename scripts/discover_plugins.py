#!/usr/bin/env python
"""
Plugin discovery script for Feluda.

This script discovers and loads Feluda plugins.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feluda.observability import get_logger
from feluda.plugins import PluginManager, PluginType, get_plugin_manager

log = get_logger(__name__)


def discover_plugins(
    plugin_dirs: Optional[List[str]] = None,
    config_file: Optional[str] = None,
    plugin_type: Optional[PluginType] = None,
) -> Dict[str, Any]:
    """
    Discover and load Feluda plugins.
    
    Args:
        plugin_dirs: Directories to search for plugins.
        config_file: Plugin configuration file.
        plugin_type: Type of plugins to load.
        
    Returns:
        A dictionary with information about the discovered plugins.
    """
    # Get the plugin manager
    plugin_manager = get_plugin_manager()
    
    # Add plugin directories
    if plugin_dirs:
        for plugin_dir in plugin_dirs:
            plugin_manager.add_plugin_dir(plugin_dir)
    
    # Load plugin configurations
    if config_file:
        plugin_manager.load_plugin_configs(config_file)
    
    # Discover plugins
    plugin_classes = plugin_manager.discover_plugins()
    
    # Load plugins
    plugins = plugin_manager.load_plugins()
    
    # Filter plugins by type
    if plugin_type:
        plugins = plugin_manager.get_plugins(plugin_type)
    
    # Collect plugin information
    plugin_info = {}
    
    for plugin_name, plugin in plugins.items():
        plugin_info[plugin_name] = {
            "info": plugin.info.to_dict(),
            "config": plugin.config.to_dict(),
        }
    
    return {
        "plugins": plugin_info,
        "plugin_count": len(plugin_info),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Discover and load Feluda plugins")
    parser.add_argument(
        "--plugin-dirs",
        type=str,
        nargs="+",
        help="Directories to search for plugins",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Plugin configuration file",
    )
    parser.add_argument(
        "--plugin-type",
        type=str,
        choices=[t.value for t in PluginType],
        help="Type of plugins to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "yaml"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Discover plugins
    plugin_type = PluginType(args.plugin_type) if args.plugin_type else None
    result = discover_plugins(args.plugin_dirs, args.config_file, plugin_type)
    
    # Print the result
    if args.output:
        with open(args.output, "w") as f:
            if args.format == "json":
                json.dump(result, f, indent=2)
            else:
                yaml.dump(result, f, default_flow_style=False)
    else:
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(yaml.dump(result, default_flow_style=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
