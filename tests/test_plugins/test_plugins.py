"""
Unit tests for the plugin system.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import pytest
import yaml

from feluda.plugins import (
    FeludaPlugin,
    PluginConfig,
    PluginInfo,
    PluginManager,
    PluginType,
    get_plugin_manager,
)
from feluda.plugins.plugin import create_plugin, load_plugin, load_plugin_class


# Define a sample plugin for testing
class SamplePlugin(FeludaPlugin):
    """Sample plugin for testing."""
    
    def __init__(self, config=None):
        """Initialize the plugin."""
        super().__init__(config)
        self._info = PluginInfo(
            name="sample_plugin",
            version="1.0.0",
            description="A sample plugin for testing",
            author="Tattle",
            plugin_type=PluginType.OTHER,
            entry_point="tests.test_plugins.test_plugins.SamplePlugin",
            dependencies=[],
        )
        self.initialized = False
        self.shutdown_called = False
    
    @property
    def info(self):
        """Get information about the plugin."""
        return self._info
    
    def initialize(self):
        """Initialize the plugin."""
        self.initialized = True
    
    def shutdown(self):
        """Shut down the plugin."""
        self.shutdown_called = True
    
    def sample_method(self):
        """Sample method for testing."""
        return "sample_result"


# Define another sample plugin for testing
class AnotherPlugin(FeludaPlugin):
    """Another plugin for testing."""
    
    def __init__(self, config=None):
        """Initialize the plugin."""
        super().__init__(config)
        self._info = PluginInfo(
            name="another_plugin",
            version="1.0.0",
            description="Another plugin for testing",
            author="Tattle",
            plugin_type=PluginType.OPERATOR,
            entry_point="tests.test_plugins.test_plugins.AnotherPlugin",
            dependencies=[],
        )
        self.initialized = False
        self.shutdown_called = False
    
    @property
    def info(self):
        """Get information about the plugin."""
        return self._info
    
    def initialize(self):
        """Initialize the plugin."""
        self.initialized = True
    
    def shutdown(self):
        """Shut down the plugin."""
        self.shutdown_called = True


class TestPluginInfo(unittest.TestCase):
    """Test cases for the PluginInfo class."""
    
    def test_plugin_info(self):
        """Test the PluginInfo class."""
        # Create a plugin info
        info = PluginInfo(
            name="sample_plugin",
            version="1.0.0",
            description="A sample plugin for testing",
            author="Tattle",
            plugin_type=PluginType.OTHER,
            entry_point="tests.test_plugins.test_plugins.SamplePlugin",
            dependencies=["dep1", "dep2"],
            homepage="https://github.com/tattle-made/feluda",
            repository="https://github.com/tattle-made/feluda",
            license="MIT",
        )
        
        # Check the attributes
        self.assertEqual(info.name, "sample_plugin")
        self.assertEqual(info.version, "1.0.0")
        self.assertEqual(info.description, "A sample plugin for testing")
        self.assertEqual(info.author, "Tattle")
        self.assertEqual(info.plugin_type, PluginType.OTHER)
        self.assertEqual(info.entry_point, "tests.test_plugins.test_plugins.SamplePlugin")
        self.assertEqual(info.dependencies, ["dep1", "dep2"])
        self.assertEqual(info.homepage, "https://github.com/tattle-made/feluda")
        self.assertEqual(info.repository, "https://github.com/tattle-made/feluda")
        self.assertEqual(info.license, "MIT")
        
        # Test to_dict
        info_dict = info.to_dict()
        self.assertEqual(info_dict["name"], "sample_plugin")
        self.assertEqual(info_dict["version"], "1.0.0")
        self.assertEqual(info_dict["description"], "A sample plugin for testing")
        self.assertEqual(info_dict["author"], "Tattle")
        self.assertEqual(info_dict["plugin_type"], PluginType.OTHER)
        self.assertEqual(info_dict["entry_point"], "tests.test_plugins.test_plugins.SamplePlugin")
        self.assertEqual(info_dict["dependencies"], ["dep1", "dep2"])
        self.assertEqual(info_dict["homepage"], "https://github.com/tattle-made/feluda")
        self.assertEqual(info_dict["repository"], "https://github.com/tattle-made/feluda")
        self.assertEqual(info_dict["license"], "MIT")
        
        # Test from_dict
        info2 = PluginInfo.from_dict(info_dict)
        self.assertEqual(info2.name, "sample_plugin")
        self.assertEqual(info2.version, "1.0.0")
        self.assertEqual(info2.description, "A sample plugin for testing")
        self.assertEqual(info2.author, "Tattle")
        self.assertEqual(info2.plugin_type, PluginType.OTHER)
        self.assertEqual(info2.entry_point, "tests.test_plugins.test_plugins.SamplePlugin")
        self.assertEqual(info2.dependencies, ["dep1", "dep2"])
        self.assertEqual(info2.homepage, "https://github.com/tattle-made/feluda")
        self.assertEqual(info2.repository, "https://github.com/tattle-made/feluda")
        self.assertEqual(info2.license, "MIT")


class TestPluginConfig(unittest.TestCase):
    """Test cases for the PluginConfig class."""
    
    def test_plugin_config(self):
        """Test the PluginConfig class."""
        # Create a plugin config
        config = PluginConfig(
            enabled=True,
            config={
                "param1": "value1",
                "param2": "value2",
            },
        )
        
        # Check the attributes
        self.assertTrue(config.enabled)
        self.assertEqual(config.config, {
            "param1": "value1",
            "param2": "value2",
        })
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertTrue(config_dict["enabled"])
        self.assertEqual(config_dict["config"], {
            "param1": "value1",
            "param2": "value2",
        })
        
        # Test from_dict
        config2 = PluginConfig.from_dict(config_dict)
        self.assertTrue(config2.enabled)
        self.assertEqual(config2.config, {
            "param1": "value1",
            "param2": "value2",
        })


class TestFeludaPlugin(unittest.TestCase):
    """Test cases for the FeludaPlugin class."""
    
    def test_plugin(self):
        """Test the FeludaPlugin class."""
        # Create a plugin
        plugin = SamplePlugin()
        
        # Check the attributes
        self.assertIsInstance(plugin.info, PluginInfo)
        self.assertEqual(plugin.info.name, "sample_plugin")
        self.assertEqual(plugin.info.version, "1.0.0")
        self.assertEqual(plugin.info.description, "A sample plugin for testing")
        self.assertEqual(plugin.info.author, "Tattle")
        self.assertEqual(plugin.info.plugin_type, PluginType.OTHER)
        self.assertEqual(plugin.info.entry_point, "tests.test_plugins.test_plugins.SamplePlugin")
        self.assertEqual(plugin.info.dependencies, [])
        
        # Test initialize
        plugin.initialize()
        self.assertTrue(plugin.initialized)
        
        # Test shutdown
        plugin.shutdown()
        self.assertTrue(plugin.shutdown_called)
        
        # Test is_enabled
        self.assertTrue(plugin.is_enabled())
        
        # Test get_config
        self.assertEqual(plugin.get_config(), {})
        
        # Test set_config
        plugin.set_config({"param1": "value1"})
        self.assertEqual(plugin.get_config(), {"param1": "value1"})
        
        # Test sample_method
        self.assertEqual(plugin.sample_method(), "sample_result")


class TestPluginFunctions(unittest.TestCase):
    """Test cases for the plugin functions."""
    
    def test_load_plugin_class(self):
        """Test the load_plugin_class function."""
        # Load the plugin class
        plugin_class = load_plugin_class("tests.test_plugins.test_plugins.SamplePlugin")
        
        # Check the plugin class
        self.assertIs(plugin_class, SamplePlugin)
        
        # Test with a non-existent plugin class
        with self.assertRaises(ImportError):
            load_plugin_class("tests.test_plugins.test_plugins.NonExistentPlugin")
        
        # Test with a non-plugin class
        with self.assertRaises(TypeError):
            load_plugin_class("tests.test_plugins.test_plugins.TestPluginFunctions")
    
    def test_create_plugin(self):
        """Test the create_plugin function."""
        # Create a plugin
        plugin = create_plugin(SamplePlugin)
        
        # Check the plugin
        self.assertIsInstance(plugin, SamplePlugin)
        
        # Create a plugin with a configuration
        config = PluginConfig(
            enabled=True,
            config={
                "param1": "value1",
                "param2": "value2",
            },
        )
        
        plugin = create_plugin(SamplePlugin, config)
        
        # Check the plugin
        self.assertIsInstance(plugin, SamplePlugin)
        self.assertTrue(plugin.is_enabled())
        self.assertEqual(plugin.get_config(), {
            "param1": "value1",
            "param2": "value2",
        })
        
        # Test with a plugin class that raises an exception
        class ExceptionPlugin(FeludaPlugin):
            def __init__(self, config=None):
                raise ValueError("Test exception")
            
            @property
            def info(self):
                return None
            
            def initialize(self):
                pass
            
            def shutdown(self):
                pass
        
        with self.assertRaises(Exception):
            create_plugin(ExceptionPlugin)
    
    def test_load_plugin(self):
        """Test the load_plugin function."""
        # Load the plugin
        plugin = load_plugin("tests.test_plugins.test_plugins.SamplePlugin")
        
        # Check the plugin
        self.assertIsInstance(plugin, SamplePlugin)
        
        # Load the plugin with a configuration
        config = PluginConfig(
            enabled=True,
            config={
                "param1": "value1",
                "param2": "value2",
            },
        )
        
        plugin = load_plugin("tests.test_plugins.test_plugins.SamplePlugin", config)
        
        # Check the plugin
        self.assertIsInstance(plugin, SamplePlugin)
        self.assertTrue(plugin.is_enabled())
        self.assertEqual(plugin.get_config(), {
            "param1": "value1",
            "param2": "value2",
        })
        
        # Test with a non-existent plugin
        with self.assertRaises(ImportError):
            load_plugin("tests.test_plugins.test_plugins.NonExistentPlugin")


class TestPluginManager(unittest.TestCase):
    """Test cases for the PluginManager class."""
    
    def setUp(self):
        """Set up the test case."""
        self.plugin_manager = PluginManager()
    
    def test_add_plugin_dir(self):
        """Test the add_plugin_dir method."""
        # Add a plugin directory
        self.plugin_manager.add_plugin_dir("plugins")
        
        # Check the plugin directories
        self.assertEqual(self.plugin_manager.plugin_dirs, ["plugins"])
        
        # Add the same plugin directory again
        self.plugin_manager.add_plugin_dir("plugins")
        
        # Check the plugin directories
        self.assertEqual(self.plugin_manager.plugin_dirs, ["plugins"])
        
        # Add another plugin directory
        self.plugin_manager.add_plugin_dir("examples/plugins")
        
        # Check the plugin directories
        self.assertEqual(self.plugin_manager.plugin_dirs, ["plugins", "examples/plugins"])
    
    def test_discover_plugins(self):
        """Test the discover_plugins method."""
        # Mock the pkgutil.iter_modules function
        with mock.patch("pkgutil.iter_modules") as mock_iter_modules:
            mock_iter_modules.return_value = [
                (None, "sample_plugin", True),
                (None, "another_plugin", True),
                (None, "not_a_package", False),
            ]
            
            # Mock the importlib.import_module function
            with mock.patch("importlib.import_module") as mock_import_module:
                # Create a mock module with a plugin class
                mock_module = mock.MagicMock()
                mock_module.SamplePlugin = SamplePlugin
                mock_module.AnotherPlugin = AnotherPlugin
                mock_module.NotAPlugin = object
                
                # Set up the mock module's dir
                mock_module.__dir__.return_value = ["SamplePlugin", "AnotherPlugin", "NotAPlugin"]
                
                # Set up the mock import_module function
                mock_import_module.return_value = mock_module
                
                # Add a plugin directory
                self.plugin_manager.add_plugin_dir("plugins")
                
                # Discover plugins
                plugin_classes = self.plugin_manager.discover_plugins()
                
                # Check the plugin classes
                self.assertEqual(len(plugin_classes), 2)
                self.assertIn("sample_plugin.SamplePlugin", plugin_classes)
                self.assertIn("another_plugin.AnotherPlugin", plugin_classes)
                self.assertIs(plugin_classes["sample_plugin.SamplePlugin"], SamplePlugin)
                self.assertIs(plugin_classes["another_plugin.AnotherPlugin"], AnotherPlugin)
    
    def test_load_plugin_configs(self):
        """Test the load_plugin_configs method."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json.dump({
                "plugins": {
                    "sample_plugin": {
                        "enabled": True,
                        "config": {
                            "param1": "value1",
                            "param2": "value2",
                        },
                    },
                    "another_plugin": {
                        "enabled": False,
                        "config": {
                            "param3": "value3",
                            "param4": "value4",
                        },
                    },
                },
            }, f)
            config_file = f.name
        
        try:
            # Load the plugin configurations
            plugin_configs = self.plugin_manager.load_plugin_configs(config_file)
            
            # Check the plugin configurations
            self.assertEqual(len(plugin_configs), 2)
            self.assertIn("sample_plugin", plugin_configs)
            self.assertIn("another_plugin", plugin_configs)
            
            # Check the sample_plugin configuration
            self.assertIsInstance(plugin_configs["sample_plugin"], PluginConfig)
            self.assertTrue(plugin_configs["sample_plugin"].enabled)
            self.assertEqual(plugin_configs["sample_plugin"].config, {
                "param1": "value1",
                "param2": "value2",
            })
            
            # Check the another_plugin configuration
            self.assertIsInstance(plugin_configs["another_plugin"], PluginConfig)
            self.assertFalse(plugin_configs["another_plugin"].enabled)
            self.assertEqual(plugin_configs["another_plugin"].config, {
                "param3": "value3",
                "param4": "value4",
            })
        
        finally:
            # Clean up
            os.unlink(config_file)
    
    def test_load_plugins(self):
        """Test the load_plugins method."""
        # Mock the discover_plugins method
        with mock.patch.object(self.plugin_manager, "discover_plugins") as mock_discover_plugins:
            mock_discover_plugins.return_value = {
                "sample_plugin": SamplePlugin,
                "another_plugin": AnotherPlugin,
            }
            
            # Load the plugins
            plugins = self.plugin_manager.load_plugins()
            
            # Check the plugins
            self.assertEqual(len(plugins), 2)
            self.assertIn("sample_plugin", plugins)
            self.assertIn("another_plugin", plugins)
            self.assertIsInstance(plugins["sample_plugin"], SamplePlugin)
            self.assertIsInstance(plugins["another_plugin"], AnotherPlugin)
            self.assertTrue(plugins["sample_plugin"].initialized)
            self.assertTrue(plugins["another_plugin"].initialized)
    
    def test_get_plugin(self):
        """Test the get_plugin method."""
        # Mock the discover_plugins method
        with mock.patch.object(self.plugin_manager, "discover_plugins") as mock_discover_plugins:
            mock_discover_plugins.return_value = {
                "sample_plugin": SamplePlugin,
                "another_plugin": AnotherPlugin,
            }
            
            # Load the plugins
            self.plugin_manager.load_plugins()
            
            # Get a plugin
            plugin = self.plugin_manager.get_plugin("sample_plugin")
            
            # Check the plugin
            self.assertIsInstance(plugin, SamplePlugin)
            
            # Get a non-existent plugin
            plugin = self.plugin_manager.get_plugin("non_existent_plugin")
            
            # Check the plugin
            self.assertIsNone(plugin)
    
    def test_get_plugins(self):
        """Test the get_plugins method."""
        # Mock the discover_plugins method
        with mock.patch.object(self.plugin_manager, "discover_plugins") as mock_discover_plugins:
            mock_discover_plugins.return_value = {
                "sample_plugin": SamplePlugin,
                "another_plugin": AnotherPlugin,
            }
            
            # Load the plugins
            self.plugin_manager.load_plugins()
            
            # Get all plugins
            plugins = self.plugin_manager.get_plugins()
            
            # Check the plugins
            self.assertEqual(len(plugins), 2)
            self.assertIn("sample_plugin", plugins)
            self.assertIn("another_plugin", plugins)
            self.assertIsInstance(plugins["sample_plugin"], SamplePlugin)
            self.assertIsInstance(plugins["another_plugin"], AnotherPlugin)
            
            # Get plugins of a specific type
            plugins = self.plugin_manager.get_plugins(PluginType.OPERATOR)
            
            # Check the plugins
            self.assertEqual(len(plugins), 1)
            self.assertIn("another_plugin", plugins)
            self.assertIsInstance(plugins["another_plugin"], AnotherPlugin)
            
            # Get plugins of a non-existent type
            plugins = self.plugin_manager.get_plugins(PluginType.VERIFIER)
            
            # Check the plugins
            self.assertEqual(len(plugins), 0)
    
    def test_unload_plugin(self):
        """Test the unload_plugin method."""
        # Mock the discover_plugins method
        with mock.patch.object(self.plugin_manager, "discover_plugins") as mock_discover_plugins:
            mock_discover_plugins.return_value = {
                "sample_plugin": SamplePlugin,
                "another_plugin": AnotherPlugin,
            }
            
            # Load the plugins
            self.plugin_manager.load_plugins()
            
            # Get the plugins
            sample_plugin = self.plugin_manager.get_plugin("sample_plugin")
            another_plugin = self.plugin_manager.get_plugin("another_plugin")
            
            # Unload a plugin
            result = self.plugin_manager.unload_plugin("sample_plugin")
            
            # Check the result
            self.assertTrue(result)
            
            # Check the plugin
            self.assertTrue(sample_plugin.shutdown_called)
            
            # Check the plugins
            self.assertIsNone(self.plugin_manager.get_plugin("sample_plugin"))
            self.assertIsInstance(self.plugin_manager.get_plugin("another_plugin"), AnotherPlugin)
            
            # Unload a non-existent plugin
            result = self.plugin_manager.unload_plugin("non_existent_plugin")
            
            # Check the result
            self.assertFalse(result)
    
    def test_unload_plugins(self):
        """Test the unload_plugins method."""
        # Mock the discover_plugins method
        with mock.patch.object(self.plugin_manager, "discover_plugins") as mock_discover_plugins:
            mock_discover_plugins.return_value = {
                "sample_plugin": SamplePlugin,
                "another_plugin": AnotherPlugin,
            }
            
            # Load the plugins
            self.plugin_manager.load_plugins()
            
            # Get the plugins
            sample_plugin = self.plugin_manager.get_plugin("sample_plugin")
            another_plugin = self.plugin_manager.get_plugin("another_plugin")
            
            # Unload all plugins
            self.plugin_manager.unload_plugins()
            
            # Check the plugins
            self.assertTrue(sample_plugin.shutdown_called)
            self.assertTrue(another_plugin.shutdown_called)
            self.assertIsNone(self.plugin_manager.get_plugin("sample_plugin"))
            self.assertIsNone(self.plugin_manager.get_plugin("another_plugin"))


class TestGlobalPluginFunctions(unittest.TestCase):
    """Test cases for the global plugin functions."""
    
    def test_get_plugin_manager(self):
        """Test the get_plugin_manager function."""
        # Get the plugin manager
        plugin_manager = get_plugin_manager()
        
        # Check the plugin manager
        self.assertIsInstance(plugin_manager, PluginManager)
        
        # Get the plugin manager again
        plugin_manager2 = get_plugin_manager()
        
        # Check that it's the same instance
        self.assertIs(plugin_manager2, plugin_manager)


if __name__ == "__main__":
    unittest.main()
