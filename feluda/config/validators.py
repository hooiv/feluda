"""
Configuration validators for Feluda.

This module provides validators for configuration values.
"""

import abc
import enum
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import jsonschema
from pydantic import BaseModel, Field

from feluda.observability import get_logger

log = get_logger(__name__)


class ConfigValidator(abc.ABC):
    """
    Base class for configuration validators.
    
    This class defines the interface for configuration validators.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a configuration value.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        pass


class SchemaValidator(ConfigValidator):
    """
    Schema validator.
    
    This class implements a validator that validates values against a JSON schema.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize a schema validator.
        
        Args:
            schema: The JSON schema.
        """
        self.schema = schema
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against a JSON schema.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        try:
            jsonschema.validate(instance=value, schema=self.schema)
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)


class TypeValidator(ConfigValidator):
    """
    Type validator.
    
    This class implements a validator that validates values against a type.
    """
    
    def __init__(self, type_: Type):
        """
        Initialize a type validator.
        
        Args:
            type_: The type to validate against.
        """
        self.type = type_
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against a type.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if isinstance(value, self.type):
            return True, None
        else:
            return False, f"Expected type {self.type.__name__}, got {type(value).__name__}"


class RangeValidator(ConfigValidator):
    """
    Range validator.
    
    This class implements a validator that validates values against a range.
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
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against a range.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(value, (int, float)):
            return False, f"Expected a number, got {type(value).__name__}"
        
        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} is less than the minimum value {self.min_value}"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} is greater than the maximum value {self.max_value}"
        
        return True, None


class RegexValidator(ConfigValidator):
    """
    Regex validator.
    
    This class implements a validator that validates values against a regular expression.
    """
    
    def __init__(self, pattern: str):
        """
        Initialize a regex validator.
        
        Args:
            pattern: The regular expression pattern.
        """
        self.pattern = pattern
        self.regex = re.compile(pattern)
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against a regular expression.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(value, str):
            return False, f"Expected a string, got {type(value).__name__}"
        
        if self.regex.match(value):
            return True, None
        else:
            return False, f"Value {value} does not match the pattern {self.pattern}"


class EnumValidator(ConfigValidator):
    """
    Enum validator.
    
    This class implements a validator that validates values against an enumeration.
    """
    
    def __init__(self, values: List[Any]):
        """
        Initialize an enum validator.
        
        Args:
            values: The enumeration values.
        """
        self.values = values
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against an enumeration.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        if value in self.values:
            return True, None
        else:
            return False, f"Value {value} is not one of {self.values}"


class CustomValidator(ConfigValidator):
    """
    Custom validator.
    
    This class implements a validator that validates values using a custom function.
    """
    
    def __init__(self, func: Callable[[Any], Tuple[bool, Optional[str]]]):
        """
        Initialize a custom validator.
        
        Args:
            func: The validation function.
        """
        self.func = func
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value using a custom function.
        
        Args:
            value: The value to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        return self.func(value)
