"""
Authorization module for Feluda.

This module provides authorization features for Feluda.
"""

import enum
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.observability import get_logger
from feluda.security.auth import User

log = get_logger(__name__)


class Permission(BaseModel):
    """
    Permission model.
    
    This class represents a permission in the system.
    """
    
    id: str = Field(..., description="The permission ID")
    name: str = Field(..., description="The permission name")
    description: Optional[str] = Field(None, description="The permission description")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the permission to a dictionary.
        
        Returns:
            A dictionary representation of the permission.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Permission":
        """
        Create a permission from a dictionary.
        
        Args:
            data: The dictionary to create the permission from.
            
        Returns:
            A permission.
        """
        return cls(**data)


class Role(BaseModel):
    """
    Role model.
    
    This class represents a role in the system.
    """
    
    id: str = Field(..., description="The role ID")
    name: str = Field(..., description="The role name")
    description: Optional[str] = Field(None, description="The role description")
    permissions: List[str] = Field(default_factory=list, description="The permissions")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the role to a dictionary.
        
        Returns:
            A dictionary representation of the role.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Role":
        """
        Create a role from a dictionary.
        
        Args:
            data: The dictionary to create the role from.
            
        Returns:
            A role.
        """
        return cls(**data)


class PermissionManager:
    """
    Permission manager.
    
    This class is responsible for managing permissions.
    """
    
    def __init__(self):
        """
        Initialize the permission manager.
        """
        self._permissions: Dict[str, Permission] = {}
        self._lock = threading.RLock()
    
    def register_permission(self, permission: Permission) -> None:
        """
        Register a permission.
        
        Args:
            permission: The permission to register.
        """
        with self._lock:
            self._permissions[permission.id] = permission
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """
        Get a permission by ID.
        
        Args:
            permission_id: The permission ID.
            
        Returns:
            The permission, or None if the permission is not found.
        """
        with self._lock:
            return self._permissions.get(permission_id)
    
    def get_permissions(self) -> Dict[str, Permission]:
        """
        Get all permissions.
        
        Returns:
            A dictionary mapping permission IDs to permissions.
        """
        with self._lock:
            return self._permissions.copy()
    
    def has_permission(self, user: User, permission_id: str) -> bool:
        """
        Check if a user has a permission.
        
        Args:
            user: The user to check.
            permission_id: The permission ID.
            
        Returns:
            True if the user has the permission, False otherwise.
        """
        # Check if the user has the wildcard permission
        if "*" in user.permissions:
            return True
        
        # Check if the user has the permission
        if permission_id in user.permissions:
            return True
        
        # Check if the user has a role with the permission
        role_manager = get_role_manager()
        
        for role_id in user.roles:
            role = role_manager.get_role(role_id)
            
            if role and (permission_id in role.permissions or "*" in role.permissions):
                return True
        
        return False


class RoleManager:
    """
    Role manager.
    
    This class is responsible for managing roles.
    """
    
    def __init__(self):
        """
        Initialize the role manager.
        """
        self._roles: Dict[str, Role] = {}
        self._lock = threading.RLock()
    
    def register_role(self, role: Role) -> None:
        """
        Register a role.
        
        Args:
            role: The role to register.
        """
        with self._lock:
            self._roles[role.id] = role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """
        Get a role by ID.
        
        Args:
            role_id: The role ID.
            
        Returns:
            The role, or None if the role is not found.
        """
        with self._lock:
            return self._roles.get(role_id)
    
    def get_roles(self) -> Dict[str, Role]:
        """
        Get all roles.
        
        Returns:
            A dictionary mapping role IDs to roles.
        """
        with self._lock:
            return self._roles.copy()
    
    def has_role(self, user: User, role_id: str) -> bool:
        """
        Check if a user has a role.
        
        Args:
            user: The user to check.
            role_id: The role ID.
            
        Returns:
            True if the user has the role, False otherwise.
        """
        return role_id in user.roles


# Global permission manager instance
_permission_manager = None
_permission_manager_lock = threading.RLock()


def get_permission_manager() -> PermissionManager:
    """
    Get the global permission manager instance.
    
    Returns:
        The global permission manager instance.
    """
    global _permission_manager
    
    with _permission_manager_lock:
        if _permission_manager is None:
            _permission_manager = PermissionManager()
            
            # Register default permissions
            _permission_manager.register_permission(
                Permission(
                    id="read",
                    name="Read",
                    description="Read access",
                )
            )
            
            _permission_manager.register_permission(
                Permission(
                    id="write",
                    name="Write",
                    description="Write access",
                )
            )
            
            _permission_manager.register_permission(
                Permission(
                    id="admin",
                    name="Admin",
                    description="Administrative access",
                )
            )
        
        return _permission_manager


# Global role manager instance
_role_manager = None
_role_manager_lock = threading.RLock()


def get_role_manager() -> RoleManager:
    """
    Get the global role manager instance.
    
    Returns:
        The global role manager instance.
    """
    global _role_manager
    
    with _role_manager_lock:
        if _role_manager is None:
            _role_manager = RoleManager()
            
            # Register default roles
            _role_manager.register_role(
                Role(
                    id="user",
                    name="User",
                    description="Regular user",
                    permissions=["read"],
                )
            )
            
            _role_manager.register_role(
                Role(
                    id="editor",
                    name="Editor",
                    description="Editor with write access",
                    permissions=["read", "write"],
                )
            )
            
            _role_manager.register_role(
                Role(
                    id="admin",
                    name="Admin",
                    description="Administrator with full access",
                    permissions=["*"],
                )
            )
        
        return _role_manager
