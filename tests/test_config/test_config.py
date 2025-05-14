"""
Unit tests for the configuration module.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import pytest
import yaml

from feluda.config import (
    ConfigManager,
    ConfigSource,
    FeludaConfig,
    get_config,
    get_config_manager,
    load_config,
)


class TestFeludaConfig(unittest.TestCase):
    """Test cases for the FeludaConfig class."""
    
    def test_default_config(self):
        """Test the default configuration."""
        config = FeludaConfig()
        
        # Check default values
        self.assertFalse(config.debug)
        self.assertEqual(config.log_level, "INFO")
        self.assertIsNone(config.log_file)
        self.assertEqual(config.api_host, "0.0.0.0")
        self.assertEqual(config.api_port, 8000)
        self.assertEqual(config.api_workers, 4)
        self.assertEqual(config.api_timeout, 60)
        self.assertEqual(config.dashboard_host, "0.0.0.0")
        self.assertEqual(config.dashboard_port, 8050)
        self.assertIsNone(config.database_url)
        self.assertEqual(config.database_pool_size, 5)
        self.assertEqual(config.database_max_overflow, 10)
        self.assertIsNone(config.cache_url)
        self.assertEqual(config.cache_ttl, 300)
        self.assertIsNone(config.queue_url)
        self.assertEqual(config.queue_name, "feluda")
        self.assertIsNone(config.storage_url)
        self.assertEqual(config.storage_bucket, "feluda")
        self.assertIsNone(config.secret_key)
        self.assertEqual(config.jwt_algorithm, "HS256")
        self.assertEqual(config.jwt_expiration, 3600)
        self.assertFalse(config.telemetry_enabled)
        self.assertIsNone(config.telemetry_url)
        self.assertFalse(config.metrics_enabled)
        self.assertIsNone(config.metrics_url)
        self.assertFalse(config.tracing_enabled)
        self.assertIsNone(config.tracing_url)
        self.assertEqual(config.plugin_dirs, [])
        self.assertIsNone(config.plugin_config_file)
        self.assertIsNone(config.operator_config_file)
        self.assertEqual(config.custom, {})
    
    def test_custom_config(self):
        """Test a custom configuration."""
        config = FeludaConfig(
            debug=True,
            log_level="DEBUG",
            log_file="logs/feluda.log",
            api_host="127.0.0.1",
            api_port=9000,
            api_workers=8,
            api_timeout=120,
            dashboard_host="127.0.0.1",
            dashboard_port=9050,
            database_url="sqlite:///data/feluda.db",
            database_pool_size=10,
            database_max_overflow=20,
            cache_url="redis://localhost:6379/0",
            cache_ttl=600,
            queue_url="amqp://guest:guest@localhost:5672/%2F",
            queue_name="feluda-queue",
            storage_url="file:///data/storage",
            storage_bucket="feluda-bucket",
            secret_key="secret",
            jwt_algorithm="RS256",
            jwt_expiration=7200,
            telemetry_enabled=True,
            telemetry_url="http://localhost:4317",
            metrics_enabled=True,
            metrics_url="http://localhost:9090",
            tracing_enabled=True,
            tracing_url="http://localhost:14268",
            plugin_dirs=["plugins", "examples/plugins"],
            plugin_config_file="config/plugins.yml",
            operator_config_file="config/operators.yml",
            custom={
                "feature_flags": {
                    "enable_ai_agents": True,
                    "enable_hardware_acceleration": False,
                },
            },
        )
        
        # Check custom values
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.log_file, "logs/feluda.log")
        self.assertEqual(config.api_host, "127.0.0.1")
        self.assertEqual(config.api_port, 9000)
        self.assertEqual(config.api_workers, 8)
        self.assertEqual(config.api_timeout, 120)
        self.assertEqual(config.dashboard_host, "127.0.0.1")
        self.assertEqual(config.dashboard_port, 9050)
        self.assertEqual(config.database_url, "sqlite:///data/feluda.db")
        self.assertEqual(config.database_pool_size, 10)
        self.assertEqual(config.database_max_overflow, 20)
        self.assertEqual(config.cache_url, "redis://localhost:6379/0")
        self.assertEqual(config.cache_ttl, 600)
        self.assertEqual(config.queue_url, "amqp://guest:guest@localhost:5672/%2F")
        self.assertEqual(config.queue_name, "feluda-queue")
        self.assertEqual(config.storage_url, "file:///data/storage")
        self.assertEqual(config.storage_bucket, "feluda-bucket")
        self.assertEqual(config.secret_key, "secret")
        self.assertEqual(config.jwt_algorithm, "RS256")
        self.assertEqual(config.jwt_expiration, 7200)
        self.assertTrue(config.telemetry_enabled)
        self.assertEqual(config.telemetry_url, "http://localhost:4317")
        self.assertTrue(config.metrics_enabled)
        self.assertEqual(config.metrics_url, "http://localhost:9090")
        self.assertTrue(config.tracing_enabled)
        self.assertEqual(config.tracing_url, "http://localhost:14268")
        self.assertEqual(config.plugin_dirs, ["plugins", "examples/plugins"])
        self.assertEqual(config.plugin_config_file, "config/plugins.yml")
        self.assertEqual(config.operator_config_file, "config/operators.yml")
        self.assertEqual(config.custom, {
            "feature_flags": {
                "enable_ai_agents": True,
                "enable_hardware_acceleration": False,
            },
        })
    
    def test_to_dict(self):
        """Test the to_dict method."""
        config = FeludaConfig(
            debug=True,
            log_level="DEBUG",
            api_port=9000,
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertTrue(config_dict["debug"])
        self.assertEqual(config_dict["log_level"], "DEBUG")
        self.assertEqual(config_dict["api_port"], 9000)
    
    def test_from_dict(self):
        """Test the from_dict method."""
        config_dict = {
            "debug": True,
            "log_level": "DEBUG",
            "api_port": 9000,
        }
        
        config = FeludaConfig.from_dict(config_dict)
        
        self.assertIsInstance(config, FeludaConfig)
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.api_port, 9000)
    
    def test_to_json(self):
        """Test the to_json method."""
        config = FeludaConfig(
            debug=True,
            log_level="DEBUG",
            api_port=9000,
        )
        
        config_json = config.to_json()
        
        self.assertIsInstance(config_json, str)
        
        config_dict = json.loads(config_json)
        self.assertTrue(config_dict["debug"])
        self.assertEqual(config_dict["log_level"], "DEBUG")
        self.assertEqual(config_dict["api_port"], 9000)
    
    def test_from_json(self):
        """Test the from_json method."""
        config_json = json.dumps({
            "debug": True,
            "log_level": "DEBUG",
            "api_port": 9000,
        })
        
        config = FeludaConfig.from_json(config_json)
        
        self.assertIsInstance(config, FeludaConfig)
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.api_port, 9000)
    
    def test_to_yaml(self):
        """Test the to_yaml method."""
        config = FeludaConfig(
            debug=True,
            log_level="DEBUG",
            api_port=9000,
        )
        
        config_yaml = config.to_yaml()
        
        self.assertIsInstance(config_yaml, str)
        
        config_dict = yaml.safe_load(config_yaml)
        self.assertTrue(config_dict["debug"])
        self.assertEqual(config_dict["log_level"], "DEBUG")
        self.assertEqual(config_dict["api_port"], 9000)
    
    def test_from_yaml(self):
        """Test the from_yaml method."""
        config_yaml = yaml.dump({
            "debug": True,
            "log_level": "DEBUG",
            "api_port": 9000,
        })
        
        config = FeludaConfig.from_yaml(config_yaml)
        
        self.assertIsInstance(config, FeludaConfig)
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.api_port, 9000)
    
    def test_update(self):
        """Test the update method."""
        config1 = FeludaConfig(
            debug=True,
            log_level="DEBUG",
            api_port=9000,
        )
        
        config2 = FeludaConfig(
            log_level="INFO",
            api_host="127.0.0.1",
            api_workers=8,
        )
        
        # Update with another config
        config3 = config1.update(config2)
        
        self.assertIsInstance(config3, FeludaConfig)
        self.assertTrue(config3.debug)
        self.assertEqual(config3.log_level, "INFO")
        self.assertEqual(config3.api_host, "127.0.0.1")
        self.assertEqual(config3.api_port, 9000)
        self.assertEqual(config3.api_workers, 8)
        
        # Update with a dict
        config4 = config1.update({
            "log_level": "WARNING",
            "api_timeout": 120,
        })
        
        self.assertIsInstance(config4, FeludaConfig)
        self.assertTrue(config4.debug)
        self.assertEqual(config4.log_level, "WARNING")
        self.assertEqual(config4.api_port, 9000)
        self.assertEqual(config4.api_timeout, 120)


class TestConfigManager(unittest.TestCase):
    """Test cases for the ConfigManager class."""
    
    def setUp(self):
        """Set up the test case."""
        self.config_manager = ConfigManager()
    
    def test_load_from_file_json(self):
        """Test loading configuration from a JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json.dump({
                "debug": True,
                "log_level": "DEBUG",
                "api_port": 9000,
            }, f)
            config_file = f.name
        
        try:
            # Load the configuration
            config = self.config_manager.load_from_file(config_file)
            
            # Check the configuration
            self.assertIsInstance(config, FeludaConfig)
            self.assertTrue(config.debug)
            self.assertEqual(config.log_level, "DEBUG")
            self.assertEqual(config.api_port, 9000)
            
            # Check the sources
            self.assertEqual(self.config_manager.get_source("debug"), ConfigSource.FILE)
            self.assertEqual(self.config_manager.get_source("log_level"), ConfigSource.FILE)
            self.assertEqual(self.config_manager.get_source("api_port"), ConfigSource.FILE)
        
        finally:
            # Clean up
            os.unlink(config_file)
    
    def test_load_from_file_yaml(self):
        """Test loading configuration from a YAML file."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml.dump({
                "debug": True,
                "log_level": "DEBUG",
                "api_port": 9000,
            }, f)
            config_file = f.name
        
        try:
            # Load the configuration
            config = self.config_manager.load_from_file(config_file)
            
            # Check the configuration
            self.assertIsInstance(config, FeludaConfig)
            self.assertTrue(config.debug)
            self.assertEqual(config.log_level, "DEBUG")
            self.assertEqual(config.api_port, 9000)
            
            # Check the sources
            self.assertEqual(self.config_manager.get_source("debug"), ConfigSource.FILE)
            self.assertEqual(self.config_manager.get_source("log_level"), ConfigSource.FILE)
            self.assertEqual(self.config_manager.get_source("api_port"), ConfigSource.FILE)
        
        finally:
            # Clean up
            os.unlink(config_file)
    
    def test_load_from_file_not_found(self):
        """Test loading configuration from a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_from_file("non_existent_file.json")
    
    def test_load_from_file_unsupported_format(self):
        """Test loading configuration from a file with an unsupported format."""
        # Create a temporary file with an unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"debug: true\nlog_level: DEBUG\napi_port: 9000\n")
            config_file = f.name
        
        try:
            # Load the configuration
            with self.assertRaises(ValueError):
                self.config_manager.load_from_file(config_file)
        
        finally:
            # Clean up
            os.unlink(config_file)
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        with mock.patch.dict(os.environ, {
            "FELUDA_DEBUG": "true",
            "FELUDA_LOG_LEVEL": "DEBUG",
            "FELUDA_API_PORT": "9000",
            "FELUDA_API_WORKERS": "8",
            "FELUDA_TELEMETRY_ENABLED": "1",
            "FELUDA_CACHE_TTL": "600",
            "FELUDA_CUSTOM__FEATURE_FLAGS__ENABLE_AI_AGENTS": "yes",
        }):
            # Load the configuration
            config = self.config_manager.load_from_env()
            
            # Check the configuration
            self.assertIsInstance(config, FeludaConfig)
            self.assertTrue(config.debug)
            self.assertEqual(config.log_level, "DEBUG")
            self.assertEqual(config.api_port, 9000)
            self.assertEqual(config.api_workers, 8)
            self.assertTrue(config.telemetry_enabled)
            self.assertEqual(config.cache_ttl, 600)
            
            # Check the sources
            self.assertEqual(self.config_manager.get_source("debug"), ConfigSource.ENVIRONMENT)
            self.assertEqual(self.config_manager.get_source("log_level"), ConfigSource.ENVIRONMENT)
            self.assertEqual(self.config_manager.get_source("api_port"), ConfigSource.ENVIRONMENT)
            self.assertEqual(self.config_manager.get_source("api_workers"), ConfigSource.ENVIRONMENT)
            self.assertEqual(self.config_manager.get_source("telemetry_enabled"), ConfigSource.ENVIRONMENT)
            self.assertEqual(self.config_manager.get_source("cache_ttl"), ConfigSource.ENVIRONMENT)
    
    def test_load_from_args(self):
        """Test loading configuration from command-line arguments."""
        # Load the configuration
        config = self.config_manager.load_from_args({
            "debug": True,
            "log_level": "DEBUG",
            "api_port": 9000,
            "api_workers": 8,
            "telemetry_enabled": True,
            "cache_ttl": 600,
        })
        
        # Check the configuration
        self.assertIsInstance(config, FeludaConfig)
        self.assertTrue(config.debug)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.api_port, 9000)
        self.assertEqual(config.api_workers, 8)
        self.assertTrue(config.telemetry_enabled)
        self.assertEqual(config.cache_ttl, 600)
        
        # Check the sources
        self.assertEqual(self.config_manager.get_source("debug"), ConfigSource.COMMAND_LINE)
        self.assertEqual(self.config_manager.get_source("log_level"), ConfigSource.COMMAND_LINE)
        self.assertEqual(self.config_manager.get_source("api_port"), ConfigSource.COMMAND_LINE)
        self.assertEqual(self.config_manager.get_source("api_workers"), ConfigSource.COMMAND_LINE)
        self.assertEqual(self.config_manager.get_source("telemetry_enabled"), ConfigSource.COMMAND_LINE)
        self.assertEqual(self.config_manager.get_source("cache_ttl"), ConfigSource.COMMAND_LINE)
    
    def test_get_config(self):
        """Test getting the configuration."""
        # Get the configuration
        config = self.config_manager.get_config()
        
        # Check the configuration
        self.assertIsInstance(config, FeludaConfig)
    
    def test_get_source(self):
        """Test getting the source of a configuration key."""
        # Load the configuration
        self.config_manager.load_from_args({
            "debug": True,
            "log_level": "DEBUG",
        })
        
        # Get the sources
        self.assertEqual(self.config_manager.get_source("debug"), ConfigSource.COMMAND_LINE)
        self.assertEqual(self.config_manager.get_source("log_level"), ConfigSource.COMMAND_LINE)
        self.assertIsNone(self.config_manager.get_source("api_port"))
    
    def test_get_sources(self):
        """Test getting the sources of all configuration keys."""
        # Load the configuration
        self.config_manager.load_from_args({
            "debug": True,
            "log_level": "DEBUG",
        })
        
        # Get the sources
        sources = self.config_manager.get_sources()
        
        # Check the sources
        self.assertIsInstance(sources, dict)
        self.assertEqual(sources["debug"], ConfigSource.COMMAND_LINE)
        self.assertEqual(sources["log_level"], ConfigSource.COMMAND_LINE)


class TestGlobalConfigFunctions(unittest.TestCase):
    """Test cases for the global configuration functions."""
    
    def test_get_config_manager(self):
        """Test getting the global configuration manager."""
        # Get the configuration manager
        config_manager = get_config_manager()
        
        # Check the configuration manager
        self.assertIsInstance(config_manager, ConfigManager)
        
        # Get the configuration manager again
        config_manager2 = get_config_manager()
        
        # Check that it's the same instance
        self.assertIs(config_manager2, config_manager)
    
    def test_get_config(self):
        """Test getting the global configuration."""
        # Get the configuration
        config = get_config()
        
        # Check the configuration
        self.assertIsInstance(config, FeludaConfig)
    
    def test_load_config(self):
        """Test loading configuration from various sources."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json.dump({
                "debug": True,
                "log_level": "DEBUG",
                "api_port": 9000,
            }, f)
            config_file = f.name
        
        try:
            # Set environment variables
            with mock.patch.dict(os.environ, {
                "FELUDA_LOG_LEVEL": "INFO",
                "FELUDA_API_HOST": "127.0.0.1",
            }):
                # Load the configuration
                config = load_config(
                    file_path=config_file,
                    env_prefix="FELUDA_",
                    args={
                        "api_workers": 8,
                        "telemetry_enabled": True,
                    },
                )
                
                # Check the configuration
                self.assertIsInstance(config, FeludaConfig)
                self.assertTrue(config.debug)  # From file
                self.assertEqual(config.log_level, "INFO")  # From env (overrides file)
                self.assertEqual(config.api_host, "127.0.0.1")  # From env
                self.assertEqual(config.api_port, 9000)  # From file
                self.assertEqual(config.api_workers, 8)  # From args
                self.assertTrue(config.telemetry_enabled)  # From args
                
                # Check the sources
                config_manager = get_config_manager()
                self.assertEqual(config_manager.get_source("debug"), ConfigSource.FILE)
                self.assertEqual(config_manager.get_source("log_level"), ConfigSource.ENVIRONMENT)
                self.assertEqual(config_manager.get_source("api_host"), ConfigSource.ENVIRONMENT)
                self.assertEqual(config_manager.get_source("api_port"), ConfigSource.FILE)
                self.assertEqual(config_manager.get_source("api_workers"), ConfigSource.COMMAND_LINE)
                self.assertEqual(config_manager.get_source("telemetry_enabled"), ConfigSource.COMMAND_LINE)
        
        finally:
            # Clean up
            os.unlink(config_file)


if __name__ == "__main__":
    unittest.main()
