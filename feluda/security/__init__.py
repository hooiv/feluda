"""
Security module for Feluda.

This module provides security features for Feluda, including authentication and authorization.
"""

from feluda.security.auth import (
    AuthBackend,
    AuthManager,
    AuthProvider,
    JWTAuthBackend,
    OAuth2AuthBackend,
    User,
    get_auth_manager,
)
from feluda.security.authorization import (
    Permission,
    PermissionManager,
    Role,
    RoleManager,
    get_permission_manager,
    get_role_manager,
)
from feluda.security.encryption import (
    EncryptionBackend,
    EncryptionManager,
    EncryptionProvider,
    get_encryption_manager,
)

__all__ = [
    "AuthBackend",
    "AuthManager",
    "AuthProvider",
    "EncryptionBackend",
    "EncryptionManager",
    "EncryptionProvider",
    "JWTAuthBackend",
    "OAuth2AuthBackend",
    "Permission",
    "PermissionManager",
    "Role",
    "RoleManager",
    "User",
    "get_auth_manager",
    "get_encryption_manager",
    "get_permission_manager",
    "get_role_manager",
]
