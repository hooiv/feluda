"""
Schema module for Feluda.

This module provides schema management for data validation.
"""

import enum
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger
from feluda.validation.validators import (
    Validator,
    SchemaValidator,
    TypeValidator,
    RangeValidator,
    RegexValidator,
    EnumValidator,
    CustomValidator,
    get_validator,
)

log = get_logger(__name__)


class SchemaType(str, enum.Enum):
    """Enum for schema types."""
    
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class SchemaField(BaseModel):
    """
    Schema field.
    
    This class represents a field in a schema.
    """
    
    name: str = Field(..., description="The field name")
    type: SchemaType = Field(..., description="The field type")
    description: Optional[str] = Field(None, description="The field description")
    required: bool = Field(True, description="Whether the field is required")
    default: Optional[Any] = Field(None, description="The field default value")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="The field constraints")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the schema field to a dictionary.
        
        Returns:
            A dictionary representation of the schema field.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaField":
        """
        Create a schema field from a dictionary.
        
        Args:
            data: The dictionary to create the schema field from.
            
        Returns:
            A schema field.
        """
        return cls(**data)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert the schema field to a JSON schema.
        
        Returns:
            A JSON schema representation of the schema field.
        """
        schema = {
            "type": self.type,
        }
        
        if self.description:
            schema["description"] = self.description
        
        if self.default is not None:
            schema["default"] = self.default
        
        # Add constraints
        for key, value in self.constraints.items():
            schema[key] = value
        
        return schema
    
    def get_validator(self) -> Validator:
        """
        Get a validator for the schema field.
        
        Returns:
            A validator.
        """
        if self.type == SchemaType.STRING:
            if "pattern" in self.constraints:
                return RegexValidator(self.constraints["pattern"])
            elif "enum" in self.constraints:
                return EnumValidator(self.constraints["enum"])
            else:
                return TypeValidator(str)
        
        elif self.type == SchemaType.INTEGER:
            if "minimum" in self.constraints or "maximum" in self.constraints:
                return RangeValidator(
                    min_value=self.constraints.get("minimum"),
                    max_value=self.constraints.get("maximum"),
                )
            else:
                return TypeValidator(int)
        
        elif self.type == SchemaType.NUMBER:
            if "minimum" in self.constraints or "maximum" in self.constraints:
                return RangeValidator(
                    min_value=self.constraints.get("minimum"),
                    max_value=self.constraints.get("maximum"),
                )
            else:
                return TypeValidator((int, float))
        
        elif self.type == SchemaType.BOOLEAN:
            return TypeValidator(bool)
        
        elif self.type == SchemaType.ARRAY:
            return TypeValidator(list)
        
        elif self.type == SchemaType.OBJECT:
            return TypeValidator(dict)
        
        elif self.type == SchemaType.NULL:
            return TypeValidator(type(None))
        
        else:
            raise ValueError(f"Unsupported schema type: {self.type}")


class Schema(BaseModel):
    """
    Schema.
    
    This class represents a schema for data validation.
    """
    
    name: str = Field(..., description="The schema name")
    version: str = Field(..., description="The schema version")
    description: Optional[str] = Field(None, description="The schema description")
    fields: Dict[str, SchemaField] = Field(default_factory=dict, description="The schema fields")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a dictionary.
        
        Returns:
            A dictionary representation of the schema.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "fields": {
                name: field.to_dict()
                for name, field in self.fields.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """
        Create a schema from a dictionary.
        
        Args:
            data: The dictionary to create the schema from.
            
        Returns:
            A schema.
        """
        fields = {
            name: SchemaField.from_dict(field)
            for name, field in data.get("fields", {}).items()
        }
        
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description"),
            fields=fields,
        )
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert the schema to a JSON schema.
        
        Returns:
            A JSON schema representation of the schema.
        """
        properties = {
            name: field.to_json_schema()
            for name, field in self.fields.items()
        }
        
        required = [
            name
            for name, field in self.fields.items()
            if field.required
        ]
        
        schema = {
            "type": "object",
            "properties": properties,
        }
        
        if required:
            schema["required"] = required
        
        if self.description:
            schema["description"] = self.description
        
        return schema
    
    def get_validator(self) -> Validator:
        """
        Get a validator for the schema.
        
        Returns:
            A validator.
        """
        return SchemaValidator(self.to_json_schema())
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against the schema.
        
        Args:
            data: The data to validate.
            
        Returns:
            A tuple of (is_valid, error_message).
        """
        validator = self.get_validator()
        return validator.validate(data)


class SchemaManager:
    """
    Schema manager.
    
    This class is responsible for managing schemas.
    """
    
    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize the schema manager.
        
        Args:
            schema_dir: The schema directory.
        """
        self.schema_dir = schema_dir or get_config().schema_dir or "schemas"
        self.schemas: Dict[str, Dict[str, Schema]] = {}
        self.lock = threading.RLock()
        
        # Load schemas from the schema directory
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """
        Load schemas from the schema directory.
        """
        with self.lock:
            # Check if the schema directory exists
            if not os.path.isdir(self.schema_dir):
                return
            
            # Load schemas from JSON files
            for filename in os.listdir(self.schema_dir):
                if not filename.endswith(".json"):
                    continue
                
                try:
                    # Load the schema
                    with open(os.path.join(self.schema_dir, filename), "r") as f:
                        schema_dict = json.load(f)
                    
                    # Create the schema
                    schema = Schema.from_dict(schema_dict)
                    
                    # Store the schema
                    if schema.name not in self.schemas:
                        self.schemas[schema.name] = {}
                    
                    self.schemas[schema.name][schema.version] = schema
                
                except Exception as e:
                    log.error(f"Error loading schema from {filename}: {e}")
    
    def register_schema(self, schema: Schema) -> None:
        """
        Register a schema.
        
        Args:
            schema: The schema to register.
        """
        with self.lock:
            # Store the schema
            if schema.name not in self.schemas:
                self.schemas[schema.name] = {}
            
            self.schemas[schema.name][schema.version] = schema
            
            # Save the schema to a file
            self._save_schema(schema)
    
    def _save_schema(self, schema: Schema) -> None:
        """
        Save a schema to a file.
        
        Args:
            schema: The schema to save.
        """
        try:
            # Create the schema directory if it doesn't exist
            os.makedirs(self.schema_dir, exist_ok=True)
            
            # Save the schema
            with open(os.path.join(self.schema_dir, f"{schema.name}_{schema.version}.json"), "w") as f:
                json.dump(schema.to_dict(), f, indent=2)
        
        except Exception as e:
            log.error(f"Error saving schema {schema.name} version {schema.version}: {e}")
    
    def get_schema(self, name: str, version: Optional[str] = None) -> Optional[Schema]:
        """
        Get a schema.
        
        Args:
            name: The schema name.
            version: The schema version. If None, get the latest version.
            
        Returns:
            The schema, or None if the schema is not found.
        """
        with self.lock:
            # Check if the schema exists
            if name not in self.schemas:
                return None
            
            # Get the schema version
            if version:
                return self.schemas[name].get(version)
            else:
                # Get the latest version
                versions = list(self.schemas[name].keys())
                
                if not versions:
                    return None
                
                latest_version = max(versions)
                return self.schemas[name][latest_version]
    
    def get_schemas(self) -> Dict[str, Dict[str, Schema]]:
        """
        Get all schemas.
        
        Returns:
            A dictionary mapping schema names to dictionaries mapping schema versions to schemas.
        """
        with self.lock:
            return self.schemas.copy()
    
    def delete_schema(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a schema.
        
        Args:
            name: The schema name.
            version: The schema version. If None, delete all versions.
            
        Returns:
            True if the schema was deleted, False otherwise.
        """
        with self.lock:
            # Check if the schema exists
            if name not in self.schemas:
                return False
            
            # Delete the schema
            if version:
                if version not in self.schemas[name]:
                    return False
                
                # Delete the schema file
                try:
                    os.remove(os.path.join(self.schema_dir, f"{name}_{version}.json"))
                except Exception as e:
                    log.error(f"Error deleting schema file for {name} version {version}: {e}")
                
                # Delete the schema
                del self.schemas[name][version]
                
                # Delete the schema name if there are no more versions
                if not self.schemas[name]:
                    del self.schemas[name]
            else:
                # Delete all schema files
                for version in self.schemas[name]:
                    try:
                        os.remove(os.path.join(self.schema_dir, f"{name}_{version}.json"))
                    except Exception as e:
                        log.error(f"Error deleting schema file for {name} version {version}: {e}")
                
                # Delete the schema
                del self.schemas[name]
            
            return True


# Global schema manager instance
_schema_manager = None
_schema_manager_lock = threading.RLock()


def get_schema_manager() -> SchemaManager:
    """
    Get the global schema manager instance.
    
    Returns:
        The global schema manager instance.
    """
    global _schema_manager
    
    with _schema_manager_lock:
        if _schema_manager is None:
            _schema_manager = SchemaManager()
        
        return _schema_manager
