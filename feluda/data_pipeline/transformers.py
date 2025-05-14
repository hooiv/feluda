"""
Transformers module for Feluda.

This module provides transformers for the data pipeline.
"""

import abc
import enum
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import pandas as pd
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class DataTransformer(abc.ABC):
    """
    Base class for data transformers.
    
    This class defines the interface for data transformers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a data transformer.
        
        Args:
            name: The transformer name.
            config: The transformer configuration.
        """
        self.name = name
        self.config = config
    
    @abc.abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform data.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        pass


class MapTransformer(DataTransformer):
    """
    Map transformer.
    
    This class implements a transformer that applies a function to each row or element.
    """
    
    def transform(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Transform data by applying a function to each row or element.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        # Get the function
        function_str = self.config.get("function")
        
        if not function_str:
            raise ValueError("Function not specified")
        
        # Compile the function
        function = eval(function_str)
        
        # Apply the function
        if isinstance(data, pd.DataFrame):
            # Apply the function to each row
            return data.apply(function, axis=1)
        else:
            # Apply the function to each element
            return [function(item) for item in data]


class FilterTransformer(DataTransformer):
    """
    Filter transformer.
    
    This class implements a transformer that filters data based on a condition.
    """
    
    def transform(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Transform data by filtering based on a condition.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        # Get the condition
        condition_str = self.config.get("condition")
        
        if not condition_str:
            raise ValueError("Condition not specified")
        
        # Compile the condition
        condition = eval(condition_str)
        
        # Apply the condition
        if isinstance(data, pd.DataFrame):
            # Filter the DataFrame
            return data[data.apply(condition, axis=1)]
        else:
            # Filter the list
            return [item for item in data if condition(item)]


class JoinTransformer(DataTransformer):
    """
    Join transformer.
    
    This class implements a transformer that joins two datasets.
    """
    
    def transform(self, data: Tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform data by joining two datasets.
        
        Args:
            data: A tuple of two DataFrames to join.
            
        Returns:
            The joined DataFrame.
        """
        # Get the join parameters
        left_on = self.config.get("left_on")
        right_on = self.config.get("right_on")
        how = self.config.get("how", "inner")
        
        if not left_on or not right_on:
            raise ValueError("Join columns not specified")
        
        # Get the DataFrames
        left_df, right_df = data
        
        # Join the DataFrames
        return pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)


class ReduceTransformer(DataTransformer):
    """
    Reduce transformer.
    
    This class implements a transformer that reduces data using a function.
    """
    
    def transform(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Any:
        """
        Transform data by reducing it using a function.
        
        Args:
            data: The data to transform.
            
        Returns:
            The reduced data.
        """
        # Get the function
        function_str = self.config.get("function")
        
        if not function_str:
            raise ValueError("Function not specified")
        
        # Compile the function
        function = eval(function_str)
        
        # Apply the function
        if isinstance(data, pd.DataFrame):
            # Group by columns
            group_by = self.config.get("group_by")
            
            if group_by:
                # Group by columns and apply the function
                return data.groupby(group_by).apply(function)
            else:
                # Apply the function to the entire DataFrame
                return function(data)
        else:
            # Apply the function to the list
            return function(data)


class TransformerManager:
    """
    Transformer manager.
    
    This class is responsible for managing data transformers.
    """
    
    def __init__(self):
        """
        Initialize the transformer manager.
        """
        self.transformers: Dict[str, DataTransformer] = {}
        self.lock = threading.RLock()
    
    def register_transformer(self, transformer: DataTransformer) -> None:
        """
        Register a transformer.
        
        Args:
            transformer: The transformer to register.
        """
        with self.lock:
            self.transformers[transformer.name] = transformer
    
    def get_transformer(self, name: str) -> Optional[DataTransformer]:
        """
        Get a transformer by name.
        
        Args:
            name: The transformer name.
            
        Returns:
            The transformer, or None if the transformer is not found.
        """
        with self.lock:
            return self.transformers.get(name)
    
    def get_transformers(self) -> Dict[str, DataTransformer]:
        """
        Get all transformers.
        
        Returns:
            A dictionary mapping transformer names to transformers.
        """
        with self.lock:
            return self.transformers.copy()
    
    def create_map_transformer(self, name: str, config: Dict[str, Any]) -> MapTransformer:
        """
        Create a map transformer.
        
        Args:
            name: The transformer name.
            config: The transformer configuration.
            
        Returns:
            The map transformer.
        """
        with self.lock:
            transformer = MapTransformer(name, config)
            self.register_transformer(transformer)
            return transformer
    
    def create_filter_transformer(self, name: str, config: Dict[str, Any]) -> FilterTransformer:
        """
        Create a filter transformer.
        
        Args:
            name: The transformer name.
            config: The transformer configuration.
            
        Returns:
            The filter transformer.
        """
        with self.lock:
            transformer = FilterTransformer(name, config)
            self.register_transformer(transformer)
            return transformer
    
    def create_join_transformer(self, name: str, config: Dict[str, Any]) -> JoinTransformer:
        """
        Create a join transformer.
        
        Args:
            name: The transformer name.
            config: The transformer configuration.
            
        Returns:
            The join transformer.
        """
        with self.lock:
            transformer = JoinTransformer(name, config)
            self.register_transformer(transformer)
            return transformer
    
    def create_reduce_transformer(self, name: str, config: Dict[str, Any]) -> ReduceTransformer:
        """
        Create a reduce transformer.
        
        Args:
            name: The transformer name.
            config: The transformer configuration.
            
        Returns:
            The reduce transformer.
        """
        with self.lock:
            transformer = ReduceTransformer(name, config)
            self.register_transformer(transformer)
            return transformer


# Global transformer manager instance
_transformer_manager = None
_transformer_manager_lock = threading.RLock()


def get_transformer_manager() -> TransformerManager:
    """
    Get the global transformer manager instance.
    
    Returns:
        The global transformer manager instance.
    """
    global _transformer_manager
    
    with _transformer_manager_lock:
        if _transformer_manager is None:
            _transformer_manager = TransformerManager()
        
        return _transformer_manager
