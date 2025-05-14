"""
Documentation renderer for Feluda.

This module provides renderers for generating documentation in different formats.
"""

import abc
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from feluda.docs.parser import AttributeDoc, ClassDoc, FunctionDoc, ModuleDoc
from feluda.observability import get_logger

log = get_logger(__name__)


class DocRenderer(abc.ABC):
    """
    Base class for documentation renderers.
    
    This class defines the interface for documentation renderers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        """
        Initialize the documentation renderer.
        """
        self.config = None
        self.lock = threading.RLock()
    
    def set_config(self, config: Any) -> None:
        """
        Set the renderer configuration.
        
        Args:
            config: The renderer configuration.
        """
        with self.lock:
            self.config = config
    
    @abc.abstractmethod
    def render(self, module_docs: List[ModuleDoc], output_dir: str) -> None:
        """
        Render documentation.
        
        Args:
            module_docs: The module documentation to render.
            output_dir: The output directory.
        """
        pass


class MarkdownRenderer(DocRenderer):
    """
    Markdown documentation renderer.
    
    This class implements a renderer for generating documentation in Markdown format.
    """
    
    def render(self, module_docs: List[ModuleDoc], output_dir: str) -> None:
        """
        Render documentation in Markdown format.
        
        Args:
            module_docs: The module documentation to render.
            output_dir: The output directory.
        """
        with self.lock:
            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the index file
            self._create_index(module_docs, output_dir)
            
            # Render each module
            for module_doc in module_docs:
                self._render_module(module_doc, output_dir)
    
    def _create_index(self, module_docs: List[ModuleDoc], output_dir: str) -> None:
        """
        Create the index file.
        
        Args:
            module_docs: The module documentation.
            output_dir: The output directory.
        """
        with open(os.path.join(output_dir, "index.md"), "w") as f:
            f.write("# API Documentation\n\n")
            
            # Write the module list
            f.write("## Modules\n\n")
            
            for module_doc in sorted(module_docs, key=lambda m: m.name):
                f.write(f"- [{module_doc.name}]({module_doc.name.replace('.', '/')}.md)\n")
    
    def _render_module(self, module_doc: ModuleDoc, output_dir: str) -> None:
        """
        Render a module.
        
        Args:
            module_doc: The module documentation.
            output_dir: The output directory.
        """
        # Create the module directory
        module_dir = os.path.join(output_dir, *module_doc.name.split(".")[:-1])
        os.makedirs(module_dir, exist_ok=True)
        
        # Create the module file
        module_file = os.path.join(output_dir, module_doc.name.replace(".", "/") + ".md")
        
        with open(module_file, "w") as f:
            # Write the module header
            f.write(f"# {module_doc.name}\n\n")
            
            # Write the module documentation
            if module_doc.doc:
                f.write(f"{module_doc.doc}\n\n")
            
            # Write the module source code
            if self.config and self.config.include_source and module_doc.source:
                f.write("## Source Code\n\n")
                f.write("```python\n")
                f.write(module_doc.source)
                f.write("\n```\n\n")
            
            # Write the class list
            if module_doc.classes:
                f.write("## Classes\n\n")
                
                for class_doc in sorted(module_doc.classes, key=lambda c: c.name):
                    # Skip private classes
                    if class_doc.is_private and self.config and not self.config.include_private:
                        continue
                    
                    f.write(f"- [{class_doc.name}](#{class_doc.name.lower()})\n")
                
                f.write("\n")
            
            # Write the function list
            if module_doc.functions:
                f.write("## Functions\n\n")
                
                for function_doc in sorted(module_doc.functions, key=lambda f: f.name):
                    # Skip private functions
                    if function_doc.is_private and self.config and not self.config.include_private:
                        continue
                    
                    f.write(f"- [{function_doc.name}](#{function_doc.name.lower()})\n")
                
                f.write("\n")
            
            # Write the class documentation
            for class_doc in sorted(module_doc.classes, key=lambda c: c.name):
                # Skip private classes
                if class_doc.is_private and self.config and not self.config.include_private:
                    continue
                
                self._render_class(class_doc, f)
            
            # Write the function documentation
            for function_doc in sorted(module_doc.functions, key=lambda f: f.name):
                # Skip private functions
                if function_doc.is_private and self.config and not self.config.include_private:
                    continue
                
                self._render_function(function_doc, f)
    
    def _render_class(self, class_doc: ClassDoc, f: Any) -> None:
        """
        Render a class.
        
        Args:
            class_doc: The class documentation.
            f: The file to write to.
        """
        # Write the class header
        f.write(f"## {class_doc.name}\n\n")
        
        # Write the class bases
        if class_doc.bases:
            f.write(f"Bases: {', '.join(class_doc.bases)}\n\n")
        
        # Write the class documentation
        if class_doc.doc:
            f.write(f"{class_doc.doc}\n\n")
        
        # Write the class source code
        if self.config and self.config.include_source and class_doc.source:
            f.write("### Source Code\n\n")
            f.write("```python\n")
            f.write(class_doc.source)
            f.write("\n```\n\n")
        
        # Write the attribute list
        if class_doc.attributes:
            f.write("### Attributes\n\n")
            
            for attribute_doc in sorted(class_doc.attributes, key=lambda a: a.name):
                # Skip private attributes
                if attribute_doc.is_private and self.config and not self.config.include_private:
                    continue
                
                # Write the attribute
                f.write(f"- **{attribute_doc.name}**")
                
                if attribute_doc.type:
                    f.write(f" ({attribute_doc.type})")
                
                if attribute_doc.default:
                    f.write(f" = {attribute_doc.default}")
                
                f.write("\n")
                
                if attribute_doc.doc:
                    f.write(f"  {attribute_doc.doc}\n")
            
            f.write("\n")
        
        # Write the method list
        if class_doc.methods:
            f.write("### Methods\n\n")
            
            for method_doc in sorted(class_doc.methods, key=lambda m: m.name):
                # Skip private methods
                if method_doc.is_private and self.config and not self.config.include_private:
                    continue
                
                # Write the method
                f.write(f"- [{method_doc.name}](#{class_doc.name.lower()}-{method_doc.name.lower()})\n")
            
            f.write("\n")
        
        # Write the method documentation
        for method_doc in sorted(class_doc.methods, key=lambda m: m.name):
            # Skip private methods
            if method_doc.is_private and self.config and not self.config.include_private:
                continue
            
            self._render_function(method_doc, f, class_name=class_doc.name)
    
    def _render_function(self, function_doc: FunctionDoc, f: Any, class_name: Optional[str] = None) -> None:
        """
        Render a function.
        
        Args:
            function_doc: The function documentation.
            f: The file to write to.
            class_name: The class name, if the function is a method.
        """
        # Write the function header
        if class_name:
            f.write(f"### {class_name}.{function_doc.name}\n\n")
        else:
            f.write(f"## {function_doc.name}\n\n")
        
        # Write the function signature
        f.write("```python\n")
        
        if function_doc.is_static:
            f.write("@staticmethod\n")
        elif function_doc.is_class:
            f.write("@classmethod\n")
        elif function_doc.is_abstract:
            f.write("@abstractmethod\n")
        
        if class_name:
            f.write(f"def {function_doc.name}{function_doc.signature}\n")
        else:
            f.write(f"def {function_doc.name}{function_doc.signature}\n")
        
        f.write("```\n\n")
        
        # Write the function documentation
        if function_doc.doc:
            f.write(f"{function_doc.doc}\n\n")
        
        # Write the function parameters
        if function_doc.parameters:
            f.write("**Parameters:**\n\n")
            
            for param in function_doc.parameters:
                f.write(f"- **{param['name']}**")
                
                if param['type']:
                    f.write(f" ({param['type']})")
                
                if param['default'] and param['default'] != "None":
                    f.write(f" = {param['default']}")
                
                f.write("\n")
            
            f.write("\n")
        
        # Write the function return type
        if function_doc.return_type or function_doc.return_doc:
            f.write("**Returns:**\n\n")
            
            if function_doc.return_type:
                f.write(f"- ({function_doc.return_type})")
                
                if function_doc.return_doc:
                    f.write(f": {function_doc.return_doc}")
                
                f.write("\n")
            elif function_doc.return_doc:
                f.write(f"- {function_doc.return_doc}\n")
            
            f.write("\n")
        
        # Write the function exceptions
        if function_doc.exceptions:
            f.write("**Raises:**\n\n")
            
            for exception in function_doc.exceptions:
                f.write(f"- **{exception['type']}**: {exception['description']}\n")
            
            f.write("\n")
        
        # Write the function examples
        if self.config and self.config.include_examples and function_doc.examples:
            f.write("**Examples:**\n\n")
            
            for example in function_doc.examples:
                f.write(f"```python\n{example}\n```\n\n")
        
        # Write the function source code
        if self.config and self.config.include_source and function_doc.source:
            f.write("**Source Code:**\n\n")
            f.write("```python\n")
            f.write(function_doc.source)
            f.write("\n```\n\n")


class HTMLRenderer(DocRenderer):
    """
    HTML documentation renderer.
    
    This class implements a renderer for generating documentation in HTML format.
    """
    
    def render(self, module_docs: List[ModuleDoc], output_dir: str) -> None:
        """
        Render documentation in HTML format.
        
        Args:
            module_docs: The module documentation to render.
            output_dir: The output directory.
        """
        # TODO: Implement HTML rendering
        pass


class SphinxRenderer(DocRenderer):
    """
    Sphinx documentation renderer.
    
    This class implements a renderer for generating documentation in Sphinx format.
    """
    
    def render(self, module_docs: List[ModuleDoc], output_dir: str) -> None:
        """
        Render documentation in Sphinx format.
        
        Args:
            module_docs: The module documentation to render.
            output_dir: The output directory.
        """
        # TODO: Implement Sphinx rendering
        pass


# Dictionary of documentation renderers
_doc_renderers: Dict[str, DocRenderer] = {
    "markdown": MarkdownRenderer(),
    "html": HTMLRenderer(),
    "sphinx": SphinxRenderer(),
}


def get_doc_renderer(format: str = "markdown") -> DocRenderer:
    """
    Get a documentation renderer.
    
    Args:
        format: The documentation format.
        
    Returns:
        A documentation renderer.
    """
    if format not in _doc_renderers:
        raise ValueError(f"Unknown documentation format: {format}")
    
    return _doc_renderers[format]
