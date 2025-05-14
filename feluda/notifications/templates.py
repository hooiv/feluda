"""
Notification templates for Feluda.

This module provides templates for notifications.
"""

import abc
import enum
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import jinja2
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class NotificationTemplate(BaseModel):
    """
    Notification template.
    
    This class represents a template for notifications.
    """
    
    name: str = Field(..., description="The template name")
    title_template: str = Field(..., description="The title template")
    message_template: str = Field(..., description="The message template")
    description: Optional[str] = Field(None, description="The template description")
    
    def render_title(self, data: Dict[str, Any]) -> str:
        """
        Render the title template.
        
        Args:
            data: The template data.
            
        Returns:
            The rendered title.
        """
        try:
            # Create the Jinja2 environment
            env = jinja2.Environment()
            
            # Compile the template
            template = env.from_string(self.title_template)
            
            # Render the template
            return template.render(**data)
        
        except Exception as e:
            log.error(f"Error rendering title template: {e}")
            return self.title_template
    
    def render_message(self, data: Dict[str, Any]) -> str:
        """
        Render the message template.
        
        Args:
            data: The template data.
            
        Returns:
            The rendered message.
        """
        try:
            # Create the Jinja2 environment
            env = jinja2.Environment()
            
            # Compile the template
            template = env.from_string(self.message_template)
            
            # Render the template
            return template.render(**data)
        
        except Exception as e:
            log.error(f"Error rendering message template: {e}")
            return self.message_template
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary.
        
        Returns:
            A dictionary representation of the template.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationTemplate":
        """
        Create a template from a dictionary.
        
        Args:
            data: The dictionary to create the template from.
            
        Returns:
            A template.
        """
        return cls(**data)


class TemplateManager:
    """
    Template manager.
    
    This class is responsible for managing notification templates.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the template manager.
        
        Args:
            template_dir: The template directory.
        """
        self.template_dir = template_dir or get_config().template_dir or "notifications/templates"
        self.templates: Dict[str, NotificationTemplate] = {}
        self.lock = threading.RLock()
        
        # Load templates from the template directory
        self._load_templates()
    
    def _load_templates(self) -> None:
        """
        Load templates from the template directory.
        """
        with self.lock:
            # Check if the template directory exists
            if not os.path.isdir(self.template_dir):
                return
            
            # Load templates from JSON files
            for filename in os.listdir(self.template_dir):
                if not filename.endswith(".json"):
                    continue
                
                try:
                    # Load the template
                    with open(os.path.join(self.template_dir, filename), "r") as f:
                        template_dict = json.load(f)
                    
                    # Create the template
                    template = NotificationTemplate.from_dict(template_dict)
                    
                    # Store the template
                    self.templates[template.name] = template
                
                except Exception as e:
                    log.error(f"Error loading template from {filename}: {e}")
    
    def register_template(self, template: NotificationTemplate) -> None:
        """
        Register a template.
        
        Args:
            template: The template to register.
        """
        with self.lock:
            self.templates[template.name] = template
            
            # Save the template to a file
            self._save_template(template)
    
    def _save_template(self, template: NotificationTemplate) -> None:
        """
        Save a template to a file.
        
        Args:
            template: The template to save.
        """
        try:
            # Create the template directory if it doesn't exist
            os.makedirs(self.template_dir, exist_ok=True)
            
            # Save the template
            with open(os.path.join(self.template_dir, f"{template.name}.json"), "w") as f:
                json.dump(template.to_dict(), f, indent=2)
        
        except Exception as e:
            log.error(f"Error saving template {template.name}: {e}")
    
    def get_template(self, name: str) -> Optional[NotificationTemplate]:
        """
        Get a template by name.
        
        Args:
            name: The template name.
            
        Returns:
            The template, or None if the template is not found.
        """
        with self.lock:
            return self.templates.get(name)
    
    def get_templates(self) -> Dict[str, NotificationTemplate]:
        """
        Get all templates.
        
        Returns:
            A dictionary mapping template names to templates.
        """
        with self.lock:
            return self.templates.copy()
    
    def delete_template(self, name: str) -> bool:
        """
        Delete a template.
        
        Args:
            name: The template name.
            
        Returns:
            True if the template was deleted, False otherwise.
        """
        with self.lock:
            if name not in self.templates:
                return False
            
            # Delete the template
            del self.templates[name]
            
            # Delete the template file
            try:
                os.remove(os.path.join(self.template_dir, f"{name}.json"))
            except Exception as e:
                log.error(f"Error deleting template file for {name}: {e}")
            
            return True


# Global template manager instance
_template_manager = None
_template_manager_lock = threading.RLock()


def get_template_manager() -> TemplateManager:
    """
    Get the global template manager instance.
    
    Returns:
        The global template manager instance.
    """
    global _template_manager
    
    with _template_manager_lock:
        if _template_manager is None:
            _template_manager = TemplateManager()
            
            # Register default templates
            _template_manager.register_template(
                NotificationTemplate(
                    name="alert",
                    title_template="Alert: {{ alert_name }}",
                    message_template="Alert: {{ alert_name }}\nLevel: {{ alert_level }}\nMessage: {{ alert_message }}",
                    description="Template for alerts",
                ),
            )
            
            _template_manager.register_template(
                NotificationTemplate(
                    name="deployment",
                    title_template="Deployment: {{ deployment_name }}",
                    message_template="Deployment: {{ deployment_name }}\nStatus: {{ deployment_status }}\nMessage: {{ deployment_message }}",
                    description="Template for deployments",
                ),
            )
            
            _template_manager.register_template(
                NotificationTemplate(
                    name="experiment",
                    title_template="Experiment: {{ experiment_name }}",
                    message_template="Experiment: {{ experiment_name }}\nStatus: {{ experiment_status }}\nMessage: {{ experiment_message }}",
                    description="Template for experiments",
                ),
            )
        
        return _template_manager
