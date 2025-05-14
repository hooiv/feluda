"""
Documentation generator for Feluda.

This module provides a documentation generator for Feluda.
"""

import enum
import importlib
import inspect
import logging
import os
import pkgutil
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.docs.parser import DocParser, ModuleDoc, get_doc_parser
from feluda.docs.renderer import DocRenderer, MarkdownRenderer, get_doc_renderer
from feluda.observability import get_logger

log = get_logger(__name__)


class DocFormat(str, enum.Enum):
    """Enum for documentation formats."""
    
    MARKDOWN = "markdown"
    HTML = "html"
    SPHINX = "sphinx"


class DocConfig(BaseModel):
    """
    Documentation configuration.
    
    This class represents the configuration for the documentation generator.
    """
    
    output_dir: str = Field(..., description="The output directory")
    format: DocFormat = Field(DocFormat.MARKDOWN, description="The documentation format")
    include_private: bool = Field(False, description="Whether to include private members")
    include_source: bool = Field(True, description="Whether to include source code")
    include_examples: bool = Field(True, description="Whether to include examples")
    include_tests: bool = Field(False, description="Whether to include tests")
    exclude_modules: List[str] = Field(default_factory=list, description="Modules to exclude")
    exclude_patterns: List[str] = Field(default_factory=list, description="Patterns to exclude")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            A dictionary representation of the configuration.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            data: The dictionary to create the configuration from.
            
        Returns:
            A configuration.
        """
        return cls(**data)


class DocGenerator:
    """
    Documentation generator.
    
    This class is responsible for generating documentation for Feluda.
    """
    
    def __init__(
        self,
        parser: Optional[DocParser] = None,
        renderer: Optional[DocRenderer] = None,
        config: Optional[DocConfig] = None,
    ):
        """
        Initialize the documentation generator.
        
        Args:
            parser: The documentation parser.
            renderer: The documentation renderer.
            config: The documentation configuration.
        """
        self.parser = parser or get_doc_parser()
        self.renderer = renderer or get_doc_renderer()
        self.config = config or DocConfig(output_dir="docs")
        self.lock = threading.RLock()
    
    def generate_docs(self, package_name: str) -> None:
        """
        Generate documentation for a package.
        
        Args:
            package_name: The package name.
        """
        with self.lock:
            # Create the output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Parse the package
            package = importlib.import_module(package_name)
            module_docs = self._parse_package(package)
            
            # Render the documentation
            self._render_docs(module_docs)
    
    def _parse_package(self, package: Any) -> List[ModuleDoc]:
        """
        Parse a package.
        
        Args:
            package: The package to parse.
            
        Returns:
            A list of module documentation.
        """
        module_docs = []
        
        # Parse the package module
        module_doc = self.parser.parse_module(package)
        module_docs.append(module_doc)
        
        # Parse submodules
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            # Check if the module should be excluded
            if any(name == excluded or name.startswith(excluded + ".") for excluded in self.config.exclude_modules):
                continue
            
            try:
                # Import the module
                module = importlib.import_module(name)
                
                # Parse the module
                module_doc = self.parser.parse_module(module)
                module_docs.append(module_doc)
                
                # Parse subpackages
                if is_pkg:
                    module_docs.extend(self._parse_package(module))
            
            except Exception as e:
                log.error(f"Error parsing module {name}: {e}")
        
        return module_docs
    
    def _render_docs(self, module_docs: List[ModuleDoc]) -> None:
        """
        Render documentation.
        
        Args:
            module_docs: The module documentation to render.
        """
        # Set the renderer configuration
        self.renderer.set_config(self.config)
        
        # Render the documentation
        self.renderer.render(module_docs, self.config.output_dir)


# Global documentation generator instance
_doc_generator = None
_doc_generator_lock = threading.RLock()


def get_doc_generator() -> DocGenerator:
    """
    Get the global documentation generator instance.
    
    Returns:
        The global documentation generator instance.
    """
    global _doc_generator
    
    with _doc_generator_lock:
        if _doc_generator is None:
            # Create the documentation configuration
            config = DocConfig(
                output_dir=get_config().doc_output_dir or "docs",
                format=DocFormat(get_config().doc_format or DocFormat.MARKDOWN),
                include_private=get_config().doc_include_private or False,
                include_source=get_config().doc_include_source or True,
                include_examples=get_config().doc_include_examples or True,
                include_tests=get_config().doc_include_tests or False,
                exclude_modules=get_config().doc_exclude_modules or [],
                exclude_patterns=get_config().doc_exclude_patterns or [],
            )
            
            # Create the documentation generator
            _doc_generator = DocGenerator(config=config)
        
        return _doc_generator
