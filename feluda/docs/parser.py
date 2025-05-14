"""
Documentation parser for Feluda.

This module provides a parser for extracting documentation from Python modules.
"""

import ast
import inspect
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.observability import get_logger

log = get_logger(__name__)


class AttributeDoc(BaseModel):
    """
    Attribute documentation.
    
    This class represents documentation for an attribute.
    """
    
    name: str = Field(..., description="The attribute name")
    type: Optional[str] = Field(None, description="The attribute type")
    doc: Optional[str] = Field(None, description="The attribute documentation")
    default: Optional[str] = Field(None, description="The attribute default value")
    is_property: bool = Field(False, description="Whether the attribute is a property")
    is_class_var: bool = Field(False, description="Whether the attribute is a class variable")
    is_private: bool = Field(False, description="Whether the attribute is private")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the attribute documentation to a dictionary.
        
        Returns:
            A dictionary representation of the attribute documentation.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeDoc":
        """
        Create attribute documentation from a dictionary.
        
        Args:
            data: The dictionary to create the attribute documentation from.
            
        Returns:
            Attribute documentation.
        """
        return cls(**data)


class FunctionDoc(BaseModel):
    """
    Function documentation.
    
    This class represents documentation for a function or method.
    """
    
    name: str = Field(..., description="The function name")
    doc: Optional[str] = Field(None, description="The function documentation")
    signature: str = Field(..., description="The function signature")
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="The function parameters")
    return_type: Optional[str] = Field(None, description="The function return type")
    return_doc: Optional[str] = Field(None, description="The function return documentation")
    exceptions: List[Dict[str, str]] = Field(default_factory=list, description="The function exceptions")
    examples: List[str] = Field(default_factory=list, description="The function examples")
    is_method: bool = Field(False, description="Whether the function is a method")
    is_static: bool = Field(False, description="Whether the function is a static method")
    is_class: bool = Field(False, description="Whether the function is a class method")
    is_abstract: bool = Field(False, description="Whether the function is an abstract method")
    is_private: bool = Field(False, description="Whether the function is private")
    source: Optional[str] = Field(None, description="The function source code")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the function documentation to a dictionary.
        
        Returns:
            A dictionary representation of the function documentation.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionDoc":
        """
        Create function documentation from a dictionary.
        
        Args:
            data: The dictionary to create the function documentation from.
            
        Returns:
            Function documentation.
        """
        return cls(**data)


class ClassDoc(BaseModel):
    """
    Class documentation.
    
    This class represents documentation for a class.
    """
    
    name: str = Field(..., description="The class name")
    doc: Optional[str] = Field(None, description="The class documentation")
    bases: List[str] = Field(default_factory=list, description="The class bases")
    attributes: List[AttributeDoc] = Field(default_factory=list, description="The class attributes")
    methods: List[FunctionDoc] = Field(default_factory=list, description="The class methods")
    is_exception: bool = Field(False, description="Whether the class is an exception")
    is_abstract: bool = Field(False, description="Whether the class is abstract")
    is_private: bool = Field(False, description="Whether the class is private")
    source: Optional[str] = Field(None, description="The class source code")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the class documentation to a dictionary.
        
        Returns:
            A dictionary representation of the class documentation.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassDoc":
        """
        Create class documentation from a dictionary.
        
        Args:
            data: The dictionary to create the class documentation from.
            
        Returns:
            Class documentation.
        """
        return cls(**data)


class ModuleDoc(BaseModel):
    """
    Module documentation.
    
    This class represents documentation for a module.
    """
    
    name: str = Field(..., description="The module name")
    doc: Optional[str] = Field(None, description="The module documentation")
    functions: List[FunctionDoc] = Field(default_factory=list, description="The module functions")
    classes: List[ClassDoc] = Field(default_factory=list, description="The module classes")
    submodules: List[str] = Field(default_factory=list, description="The module submodules")
    source: Optional[str] = Field(None, description="The module source code")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the module documentation to a dictionary.
        
        Returns:
            A dictionary representation of the module documentation.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleDoc":
        """
        Create module documentation from a dictionary.
        
        Args:
            data: The dictionary to create the module documentation from.
            
        Returns:
            Module documentation.
        """
        return cls(**data)


class DocParser:
    """
    Documentation parser.
    
    This class is responsible for parsing documentation from Python modules.
    """
    
    def __init__(self):
        """
        Initialize the documentation parser.
        """
        self.lock = threading.RLock()
    
    def parse_module(self, module: Any) -> ModuleDoc:
        """
        Parse a module.
        
        Args:
            module: The module to parse.
            
        Returns:
            The module documentation.
        """
        with self.lock:
            # Get the module name and documentation
            name = module.__name__
            doc = inspect.getdoc(module)
            
            # Get the module source code
            try:
                source = inspect.getsource(module)
            except (TypeError, OSError):
                source = None
            
            # Create the module documentation
            module_doc = ModuleDoc(
                name=name,
                doc=doc,
                source=source,
            )
            
            # Parse the module members
            for member_name, member in inspect.getmembers(module):
                # Skip private members
                if member_name.startswith("_") and member_name != "__init__":
                    continue
                
                # Parse classes
                if inspect.isclass(member) and member.__module__ == name:
                    class_doc = self.parse_class(member)
                    module_doc.classes.append(class_doc)
                
                # Parse functions
                elif inspect.isfunction(member) and member.__module__ == name:
                    function_doc = self.parse_function(member)
                    module_doc.functions.append(function_doc)
                
                # Parse submodules
                elif inspect.ismodule(member) and member.__name__.startswith(name + "."):
                    module_doc.submodules.append(member.__name__)
            
            return module_doc
    
    def parse_class(self, cls: Type) -> ClassDoc:
        """
        Parse a class.
        
        Args:
            cls: The class to parse.
            
        Returns:
            The class documentation.
        """
        with self.lock:
            # Get the class name and documentation
            name = cls.__name__
            doc = inspect.getdoc(cls)
            
            # Get the class source code
            try:
                source = inspect.getsource(cls)
            except (TypeError, OSError):
                source = None
            
            # Create the class documentation
            class_doc = ClassDoc(
                name=name,
                doc=doc,
                bases=[base.__name__ for base in cls.__bases__ if base is not object],
                is_exception=issubclass(cls, Exception),
                is_abstract=inspect.isabstract(cls),
                is_private=name.startswith("_"),
                source=source,
            )
            
            # Parse the class members
            for member_name, member in inspect.getmembers(cls):
                # Skip special methods
                if member_name.startswith("__") and member_name.endswith("__"):
                    continue
                
                # Parse methods
                if inspect.isfunction(member) or inspect.ismethod(member):
                    method_doc = self.parse_function(member, is_method=True)
                    
                    # Check if the method is static or class method
                    if isinstance(cls.__dict__.get(member_name), staticmethod):
                        method_doc.is_static = True
                    elif isinstance(cls.__dict__.get(member_name), classmethod):
                        method_doc.is_class = True
                    
                    class_doc.methods.append(method_doc)
                
                # Parse properties
                elif isinstance(cls.__dict__.get(member_name), property):
                    property_obj = cls.__dict__[member_name]
                    
                    attribute_doc = AttributeDoc(
                        name=member_name,
                        doc=inspect.getdoc(property_obj),
                        is_property=True,
                        is_private=member_name.startswith("_"),
                    )
                    
                    class_doc.attributes.append(attribute_doc)
                
                # Parse attributes
                elif not callable(member) and not member_name.startswith("__"):
                    attribute_doc = AttributeDoc(
                        name=member_name,
                        type=type(member).__name__,
                        default=repr(member),
                        is_class_var=True,
                        is_private=member_name.startswith("_"),
                    )
                    
                    class_doc.attributes.append(attribute_doc)
            
            return class_doc
    
    def parse_function(self, func: Any, is_method: bool = False) -> FunctionDoc:
        """
        Parse a function.
        
        Args:
            func: The function to parse.
            is_method: Whether the function is a method.
            
        Returns:
            The function documentation.
        """
        with self.lock:
            # Get the function name and documentation
            name = func.__name__
            doc = inspect.getdoc(func)
            
            # Get the function signature
            signature = str(inspect.signature(func))
            
            # Get the function source code
            try:
                source = inspect.getsource(func)
            except (TypeError, OSError):
                source = None
            
            # Create the function documentation
            function_doc = FunctionDoc(
                name=name,
                doc=doc,
                signature=signature,
                is_method=is_method,
                is_abstract=getattr(func, "__isabstractmethod__", False),
                is_private=name.startswith("_") and name != "__init__",
                source=source,
            )
            
            # Parse the function parameters
            for param_name, param in inspect.signature(func).parameters.items():
                if param_name == "self" and is_method:
                    continue
                
                parameter = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation is not inspect.Parameter.empty else None,
                    "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
                    "kind": str(param.kind),
                }
                
                function_doc.parameters.append(parameter)
            
            # Parse the function return type
            return_annotation = inspect.signature(func).return_annotation
            
            if return_annotation is not inspect.Signature.empty:
                function_doc.return_type = str(return_annotation)
            
            # Parse the function documentation
            if doc:
                # Parse return documentation
                return_match = re.search(r"Returns:\s*(.*?)(?:\n\n|\Z)", doc, re.DOTALL)
                
                if return_match:
                    function_doc.return_doc = return_match.group(1).strip()
                
                # Parse exception documentation
                exception_matches = re.finditer(r"Raises:\s*(.*?)(?:\n\n|\Z)", doc, re.DOTALL)
                
                for match in exception_matches:
                    exception_text = match.group(1).strip()
                    exception_parts = exception_text.split(":", 1)
                    
                    if len(exception_parts) == 2:
                        exception_type = exception_parts[0].strip()
                        exception_desc = exception_parts[1].strip()
                        
                        function_doc.exceptions.append({
                            "type": exception_type,
                            "description": exception_desc,
                        })
                
                # Parse example documentation
                example_match = re.search(r"Examples?:\s*(.*?)(?:\n\n|\Z)", doc, re.DOTALL)
                
                if example_match:
                    examples = example_match.group(1).strip().split("\n\n")
                    function_doc.examples = [example.strip() for example in examples]
            
            return function_doc


# Global documentation parser instance
_doc_parser = None
_doc_parser_lock = threading.RLock()


def get_doc_parser() -> DocParser:
    """
    Get the global documentation parser instance.
    
    Returns:
        The global documentation parser instance.
    """
    global _doc_parser
    
    with _doc_parser_lock:
        if _doc_parser is None:
            _doc_parser = DocParser()
        
        return _doc_parser
