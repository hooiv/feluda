"""
Data validators for Feluda.

This module provides validators for data validation.
"""

import abc
import enum
import json
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import jsonschema
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class Validator(abc.ABC):
    """
    Base class for validators.
    
    This class defines the interface for validators.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        pass


class SchemaValidator(Validator):
    """
    Schema validator.
    
    This class implements a validator that validates data against a JSON schema.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize a schema validator.
        
        Args:
            schema: The JSON schema.
        """
        self.schema = schema
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a JSON schema.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)


class TypeValidator(Validator):
    """
    Type validator.
    
    This class implements a validator that validates data against a type.
    """
    
    def __init__(self, type_: Type):
        """
        Initialize a type validator.
        
        Args:
            type_: The type to validate against.
        """
        self.type = type_
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a type.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if isinstance(data, self.type):
            return True, None
        else:
            return False, f"Expected type {self.type.__name__}, got {type(data).__name__}"


class RangeValidator(Validator):
    """
    Range validator.
    
    This class implements a validator that validates data against a range.
    """
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        """
        Initialize a range validator.
        
        Args:
            min_value: The minimum value.
            max_value: The maximum value.
        """
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a range.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(data, (int, float)):
            return False, f"Expected a number, got {type(data).__name__}"
        
        if self.min_value is not None and data < self.min_value:
            return False, f"Value {data} is less than the minimum value {self.min_value}"
        
        if self.max_value is not None and data > self.max_value:
            return False, f"Value {data} is greater than the maximum value {self.max_value}"
        
        return True, None


class RegexValidator(Validator):
    """
    Regex validator.
    
    This class implements a validator that validates data against a regular expression.
    """
    
    def __init__(self, pattern: str):
        """
        Initialize a regex validator.
        
        Args:
            pattern: The regular expression pattern.
        """
        self.pattern = pattern
        self.regex = re.compile(pattern)
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a regular expression.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(data, str):
            return False, f"Expected a string, got {type(data).__name__}"
        
        if self.regex.match(data):
            return True, None
        else:
            return False, f"Value {data} does not match the pattern {self.pattern}"


class EnumValidator(Validator):
    """
    Enum validator.
    
    This class implements a validator that validates data against an enumeration.
    """
    
    def __init__(self, values: List[Any]):
        """
        Initialize an enum validator.
        
        Args:
            values: The enumeration values.
        """
        self.values = values
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against an enumeration.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if data in self.values:
            return True, None
        else:
            return False, f"Value {data} is not one of {self.values}"


class CustomValidator(Validator):
    """
    Custom validator.
    
    This class implements a validator that validates data using a custom function.
    """
    
    def __init__(self, func: Callable[[Any], Tuple[bool, Optional[str]]]):
        """
        Initialize a custom validator.
        
        Args:
            func: The validation function.
        """
        self.func = func
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data using a custom function.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        return self.func(data)


class DataFrameValidator(Validator):
    """
    DataFrame validator.
    
    This class implements a validator that validates a pandas DataFrame.
    """
    
    def __init__(
        self,
        column_validators: Dict[str, List[Validator]],
        required_columns: Optional[List[str]] = None,
    ):
        """
        Initialize a DataFrame validator.
        
        Args:
            column_validators: A dictionary mapping column names to validators.
            required_columns: A list of required columns.
        """
        self.column_validators = column_validators
        self.required_columns = required_columns or []
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a pandas DataFrame.
        
        Args:
            data: The DataFrame to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(data, pd.DataFrame):
            return False, f"Expected a pandas DataFrame, got {type(data).__name__}"
        
        # Check required columns
        for column in self.required_columns:
            if column not in data.columns:
                return False, f"Required column {column} is missing"
        
        # Validate each column
        for column, validators in self.column_validators.items():
            if column not in data.columns:
                continue
            
            for validator in validators:
                for value in data[column]:
                    is_valid, error_message = validator.validate(value)
                    
                    if not is_valid:
                        return False, f"Column {column}: {error_message}"
        
        return True, None


class ArrayValidator(Validator):
    """
    Array validator.
    
    This class implements a validator that validates a numpy array.
    """
    
    def __init__(
        self,
        shape: Optional[Tuple[Optional[int], ...]] = None,
        dtype: Optional[Type] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """
        Initialize an array validator.
        
        Args:
            shape: The expected array shape.
            dtype: The expected array dtype.
            min_value: The minimum value.
            max_value: The maximum value.
        """
        self.shape = shape
        self.dtype = dtype
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a numpy array.
        
        Args:
            data: The array to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(data, np.ndarray):
            return False, f"Expected a numpy array, got {type(data).__name__}"
        
        # Check shape
        if self.shape is not None:
            if len(data.shape) != len(self.shape):
                return False, f"Expected array with {len(self.shape)} dimensions, got {len(data.shape)}"
            
            for i, (expected, actual) in enumerate(zip(self.shape, data.shape)):
                if expected is not None and expected != actual:
                    return False, f"Expected shape {self.shape}, got {data.shape}"
        
        # Check dtype
        if self.dtype is not None and data.dtype != self.dtype:
            return False, f"Expected dtype {self.dtype}, got {data.dtype}"
        
        # Check range
        if self.min_value is not None and np.any(data < self.min_value):
            return False, f"Array contains values less than the minimum value {self.min_value}"
        
        if self.max_value is not None and np.any(data > self.max_value):
            return False, f"Array contains values greater than the maximum value {self.max_value}"
        
        return True, None


# Global validator instance
_validator = None
_validator_lock = threading.RLock()


def get_validator() -> Dict[str, Type[Validator]]:
    """
    Get the global validator instance.
    
    Returns:
        A dictionary mapping validator names to validator classes.
    """
    return {
        "schema": SchemaValidator,
        "type": TypeValidator,
        "range": RangeValidator,
        "regex": RegexValidator,
        "enum": EnumValidator,
        "custom": CustomValidator,
        "dataframe": DataFrameValidator,
        "array": ArrayValidator,
    }
