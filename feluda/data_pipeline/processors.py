"""
Processors module for Feluda.

This module provides processors for the data pipeline.
"""

import abc
import enum
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union

import pandas as pd
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.data_pipeline.connectors import Connector, get_connector_manager
from feluda.data_pipeline.transformers import DataTransformer, get_transformer_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class DataProcessor(abc.ABC):
    """
    Base class for data processors.
    
    This class defines the interface for data processors.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a data processor.
        
        Args:
            name: The processor name.
            config: The processor configuration.
        """
        self.name = name
        self.config = config
    
    @abc.abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process data.
        
        Args:
            data: The data to process.
            
        Returns:
            The processed data.
        """
        pass


class BatchProcessor(DataProcessor):
    """
    Batch processor.
    
    This class implements a processor that processes data in batches.
    """
    
    def process(self, data: Any) -> Any:
        """
        Process data in batches.
        
        Args:
            data: The data to process.
            
        Returns:
            The processed data.
        """
        # Get the batch size
        batch_size = self.config.get("batch_size", 1000)
        
        # Get the transformers
        transformer_names = self.config.get("transformers", [])
        transformer_manager = get_transformer_manager()
        transformers = [
            transformer_manager.get_transformer(name)
            for name in transformer_names
        ]
        
        # Process the data in batches
        if isinstance(data, pd.DataFrame):
            # Process a DataFrame in batches
            result = []
            
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                
                # Apply the transformers
                for transformer in transformers:
                    if transformer:
                        batch = transformer.transform(batch)
                
                result.append(batch)
            
            # Concatenate the results
            return pd.concat(result) if result else pd.DataFrame()
        
        elif isinstance(data, list):
            # Process a list in batches
            result = []
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Apply the transformers
                for transformer in transformers:
                    if transformer:
                        batch = transformer.transform(batch)
                
                result.extend(batch)
            
            return result
        
        else:
            # Process a single item
            result = data
            
            # Apply the transformers
            for transformer in transformers:
                if transformer:
                    result = transformer.transform(result)
            
            return result


class StreamProcessor(DataProcessor):
    """
    Stream processor.
    
    This class implements a processor that processes data in a stream.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a stream processor.
        
        Args:
            name: The processor name.
            config: The processor configuration.
        """
        super().__init__(name, config)
        self.running = False
        self.thread = None
    
    def process(self, data: Iterator[Any]) -> Iterator[Any]:
        """
        Process data in a stream.
        
        Args:
            data: The data to process.
            
        Returns:
            The processed data.
        """
        # Get the transformers
        transformer_names = self.config.get("transformers", [])
        transformer_manager = get_transformer_manager()
        transformers = [
            transformer_manager.get_transformer(name)
            for name in transformer_names
        ]
        
        # Process the data in a stream
        for item in data:
            result = item
            
            # Apply the transformers
            for transformer in transformers:
                if transformer:
                    result = transformer.transform(result)
            
            yield result
    
    def start(self, input_connector_name: str, output_connector_name: Optional[str] = None) -> None:
        """
        Start processing data from an input connector.
        
        Args:
            input_connector_name: The input connector name.
            output_connector_name: The output connector name.
        """
        with threading.RLock():
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(
                target=self._process_stream,
                args=(input_connector_name, output_connector_name),
            )
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop processing data.
        """
        with threading.RLock():
            if not self.running:
                return
            
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def _process_stream(self, input_connector_name: str, output_connector_name: Optional[str] = None) -> None:
        """
        Process data from an input connector.
        
        Args:
            input_connector_name: The input connector name.
            output_connector_name: The output connector name.
        """
        try:
            # Get the connectors
            connector_manager = get_connector_manager()
            input_connector = connector_manager.get_connector(input_connector_name)
            output_connector = connector_manager.get_connector(output_connector_name) if output_connector_name else None
            
            if not input_connector:
                log.error(f"Input connector {input_connector_name} not found")
                return
            
            # Read data from the input connector
            data = input_connector.read()
            
            # Process the data
            processed_data = self.process(data)
            
            # Write data to the output connector
            if output_connector:
                for item in processed_data:
                    if not self.running:
                        break
                    
                    output_connector.write(item)
            
        except Exception as e:
            log.error(f"Error processing stream: {e}")


class ProcessorManager:
    """
    Processor manager.
    
    This class is responsible for managing data processors.
    """
    
    def __init__(self):
        """
        Initialize the processor manager.
        """
        self.processors: Dict[str, DataProcessor] = {}
        self.lock = threading.RLock()
    
    def register_processor(self, processor: DataProcessor) -> None:
        """
        Register a processor.
        
        Args:
            processor: The processor to register.
        """
        with self.lock:
            self.processors[processor.name] = processor
    
    def get_processor(self, name: str) -> Optional[DataProcessor]:
        """
        Get a processor by name.
        
        Args:
            name: The processor name.
            
        Returns:
            The processor, or None if the processor is not found.
        """
        with self.lock:
            return self.processors.get(name)
    
    def get_processors(self) -> Dict[str, DataProcessor]:
        """
        Get all processors.
        
        Returns:
            A dictionary mapping processor names to processors.
        """
        with self.lock:
            return self.processors.copy()
    
    def create_batch_processor(self, name: str, config: Dict[str, Any]) -> BatchProcessor:
        """
        Create a batch processor.
        
        Args:
            name: The processor name.
            config: The processor configuration.
            
        Returns:
            The batch processor.
        """
        with self.lock:
            processor = BatchProcessor(name, config)
            self.register_processor(processor)
            return processor
    
    def create_stream_processor(self, name: str, config: Dict[str, Any]) -> StreamProcessor:
        """
        Create a stream processor.
        
        Args:
            name: The processor name.
            config: The processor configuration.
            
        Returns:
            The stream processor.
        """
        with self.lock:
            processor = StreamProcessor(name, config)
            self.register_processor(processor)
            return processor
    
    def start_processor(self, name: str, input_connector_name: str, output_connector_name: Optional[str] = None) -> bool:
        """
        Start a processor.
        
        Args:
            name: The processor name.
            input_connector_name: The input connector name.
            output_connector_name: The output connector name.
            
        Returns:
            True if the processor was started, False otherwise.
        """
        with self.lock:
            processor = self.get_processor(name)
            
            if not processor:
                return False
            
            if isinstance(processor, StreamProcessor):
                processor.start(input_connector_name, output_connector_name)
                return True
            
            return False
    
    def stop_processor(self, name: str) -> bool:
        """
        Stop a processor.
        
        Args:
            name: The processor name.
            
        Returns:
            True if the processor was stopped, False otherwise.
        """
        with self.lock:
            processor = self.get_processor(name)
            
            if not processor:
                return False
            
            if isinstance(processor, StreamProcessor):
                processor.stop()
                return True
            
            return False
    
    def process_data(self, name: str, data: Any) -> Any:
        """
        Process data using a processor.
        
        Args:
            name: The processor name.
            data: The data to process.
            
        Returns:
            The processed data.
        """
        with self.lock:
            processor = self.get_processor(name)
            
            if not processor:
                raise ValueError(f"Processor {name} not found")
            
            return processor.process(data)


# Global processor manager instance
_processor_manager = None
_processor_manager_lock = threading.RLock()


def get_processor_manager() -> ProcessorManager:
    """
    Get the global processor manager instance.
    
    Returns:
        The global processor manager instance.
    """
    global _processor_manager
    
    with _processor_manager_lock:
        if _processor_manager is None:
            _processor_manager = ProcessorManager()
        
        return _processor_manager
