"""
Sample plugin for Feluda.

This module provides a sample plugin for Feluda.
"""

from typing import Any, Dict, List, Optional

from feluda.observability import get_logger
from feluda.plugins import FeludaPlugin, PluginConfig, PluginInfo, PluginType

log = get_logger(__name__)


class SamplePlugin(FeludaPlugin):
    """
    Sample plugin for Feluda.
    
    This plugin demonstrates how to create a Feluda plugin.
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize the plugin.
        
        Args:
            config: The plugin configuration.
        """
        super().__init__(config)
        self._info = PluginInfo(
            name="sample_plugin",
            version="1.0.0",
            description="A sample plugin for Feluda",
            author="Tattle",
            plugin_type=PluginType.OTHER,
            entry_point="examples.plugins.sample_plugin.SamplePlugin",
            dependencies=[],
            homepage="https://github.com/tattle-made/feluda",
            repository="https://github.com/tattle-made/feluda",
            license="MIT",
        )
    
    @property
    def info(self) -> PluginInfo:
        """
        Get information about the plugin.
        
        Returns:
            Information about the plugin.
        """
        return self._info
    
    def initialize(self) -> None:
        """
        Initialize the plugin.
        
        This method is called when the plugin is loaded.
        """
        log.info(f"Initializing {self.info.name} v{self.info.version}")
        
        # Get the plugin configuration
        config = self.get_config()
        
        # Log the configuration
        log.info(f"Plugin configuration: {config}")
    
    def shutdown(self) -> None:
        """
        Shut down the plugin.
        
        This method is called when the plugin is unloaded.
        """
        log.info(f"Shutting down {self.info.name} v{self.info.version}")
    
    def hello(self, name: str) -> str:
        """
        Say hello to someone.
        
        Args:
            name: The name to say hello to.
            
        Returns:
            A greeting message.
        """
        return f"Hello, {name}! This is {self.info.name} v{self.info.version}."


class SampleOperatorPlugin(FeludaPlugin):
    """
    Sample operator plugin for Feluda.
    
    This plugin demonstrates how to create a Feluda operator plugin.
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize the plugin.
        
        Args:
            config: The plugin configuration.
        """
        super().__init__(config)
        self._info = PluginInfo(
            name="sample_operator_plugin",
            version="1.0.0",
            description="A sample operator plugin for Feluda",
            author="Tattle",
            plugin_type=PluginType.OPERATOR,
            entry_point="examples.plugins.sample_plugin.SampleOperatorPlugin",
            dependencies=[],
            homepage="https://github.com/tattle-made/feluda",
            repository="https://github.com/tattle-made/feluda",
            license="MIT",
        )
        self._operator = None
    
    @property
    def info(self) -> PluginInfo:
        """
        Get information about the plugin.
        
        Returns:
            Information about the plugin.
        """
        return self._info
    
    def initialize(self) -> None:
        """
        Initialize the plugin.
        
        This method is called when the plugin is loaded.
        """
        log.info(f"Initializing {self.info.name} v{self.info.version}")
        
        # Get the plugin configuration
        config = self.get_config()
        
        # Log the configuration
        log.info(f"Plugin configuration: {config}")
        
        # Create the operator
        from feluda.base_operator import BaseFeludaOperator
        from feluda.models.data_models import MediaContent, MediaMetadata, MediaType, OperatorResult
        
        class SampleOperator(BaseFeludaOperator[MediaContent, Dict[str, Any], Dict[str, Any]]):
            """
            Sample operator for Feluda.
            
            This operator demonstrates how to create a Feluda operator.
            """
            
            name = "SampleOperator"
            description = "A sample operator for Feluda"
            version = "1.0.0"
            parameters_model = Dict[str, Any]
            
            def _initialize(self) -> None:
                """Initialize the operator."""
                log.info("Initializing SampleOperator")
                
                # Set default parameters if not provided
                if not self.parameters:
                    self.parameters = {
                        "param1": "value1",
                        "param2": "value2",
                    }
            
            def _validate_input(self, input_data: MediaContent) -> bool:
                """Validate the input data."""
                if not isinstance(input_data, MediaContent):
                    return False
                
                return True
            
            def _execute(self, input_data: MediaContent) -> Dict[str, Any]:
                """Execute the operator on the input data."""
                log.info(f"Processing media: {input_data.metadata.media_id}")
                
                # Process the media
                result = {
                    "media_id": input_data.metadata.media_id,
                    "media_type": input_data.metadata.media_type.value,
                    "parameters": self.parameters,
                }
                
                return result
        
        self._operator = SampleOperator(parameters=config.get("operator_parameters", {}))
    
    def shutdown(self) -> None:
        """
        Shut down the plugin.
        
        This method is called when the plugin is unloaded.
        """
        log.info(f"Shutting down {self.info.name} v{self.info.version}")
        self._operator = None
    
    def get_operator(self) -> Any:
        """
        Get the operator.
        
        Returns:
            The operator.
        """
        return self._operator
