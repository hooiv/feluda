"""
Authentication module for Feluda.

This module provides authentication features for Feluda.
"""

import abc
import datetime
import enum
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import jwt
import requests
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class User(BaseModel):
    """
    User model.
    
    This class represents a user in the system.
    """
    
    id: str = Field(..., description="The user ID")
    username: str = Field(..., description="The username")
    email: Optional[str] = Field(None, description="The email address")
    first_name: Optional[str] = Field(None, description="The first name")
    last_name: Optional[str] = Field(None, description="The last name")
    roles: List[str] = Field(default_factory=list, description="The roles")
    permissions: List[str] = Field(default_factory=list, description="The permissions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the user to a dictionary.
        
        Returns:
            A dictionary representation of the user.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """
        Create a user from a dictionary.
        
        Args:
            data: The dictionary to create the user from.
            
        Returns:
            A user.
        """
        return cls(**data)


class AuthProvider(str, enum.Enum):
    """Enum for authentication providers."""
    
    JWT = "jwt"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    SAML = "saml"
    CUSTOM = "custom"


class AuthBackend(abc.ABC):
    """
    Base class for authentication backends.
    
    This class defines the interface for authentication backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @property
    @abc.abstractmethod
    def provider(self) -> AuthProvider:
        """
        Get the authentication provider.
        
        Returns:
            The authentication provider.
        """
        pass
    
    @abc.abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            credentials: The credentials to authenticate with.
            
        Returns:
            The authenticated user, or None if authentication failed.
        """
        pass
    
    @abc.abstractmethod
    def generate_token(self, user: User) -> str:
        """
        Generate an authentication token for a user.
        
        Args:
            user: The user to generate a token for.
            
        Returns:
            The authentication token.
        """
        pass
    
    @abc.abstractmethod
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate an authentication token.
        
        Args:
            token: The token to validate.
            
        Returns:
            The authenticated user, or None if the token is invalid.
        """
        pass


class JWTAuthBackend(AuthBackend):
    """
    JWT authentication backend.
    
    This class implements authentication using JSON Web Tokens (JWT).
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: Optional[str] = None,
        expiration: Optional[int] = None,
    ):
        """
        Initialize the JWT authentication backend.
        
        Args:
            secret_key: The secret key to use for signing tokens.
            algorithm: The algorithm to use for signing tokens.
            expiration: The token expiration time in seconds.
        """
        config = get_config()
        self._secret_key = secret_key or config.secret_key
        self._algorithm = algorithm or config.jwt_algorithm
        self._expiration = expiration or config.jwt_expiration
        
        if not self._secret_key:
            raise ValueError("Secret key is required for JWT authentication")
    
    @property
    def provider(self) -> AuthProvider:
        """
        Get the authentication provider.
        
        Returns:
            The authentication provider.
        """
        return AuthProvider.JWT
    
    def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            credentials: The credentials to authenticate with.
            
        Returns:
            The authenticated user, or None if authentication failed.
        """
        # In a real implementation, this would validate the credentials against a database
        # For now, we just return a dummy user
        if credentials.get("username") == "admin" and credentials.get("password") == "admin":
            return User(
                id="1",
                username="admin",
                email="admin@example.com",
                first_name="Admin",
                last_name="User",
                roles=["admin"],
                permissions=["*"],
            )
        
        return None
    
    def generate_token(self, user: User) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user: The user to generate a token for.
            
        Returns:
            The JWT token.
        """
        payload = {
            "sub": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=self._expiration),
        }
        
        return jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate a JWT token.
        
        Args:
            token: The token to validate.
            
        Returns:
            The authenticated user, or None if the token is invalid.
        """
        try:
            payload = jwt.decode(token, self._secret_key, algorithms=[self._algorithm])
            
            return User(
                id=payload["sub"],
                username=payload["username"],
                roles=payload["roles"],
                permissions=payload["permissions"],
            )
        
        except jwt.PyJWTError as e:
            log.error(f"Failed to validate JWT token: {e}")
            return None


class OAuth2AuthBackend(AuthBackend):
    """
    OAuth2 authentication backend.
    
    This class implements authentication using OAuth2.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        authorize_url: Optional[str] = None,
        token_url: Optional[str] = None,
        userinfo_url: Optional[str] = None,
        scope: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        """
        Initialize the OAuth2 authentication backend.
        
        Args:
            client_id: The OAuth2 client ID.
            client_secret: The OAuth2 client secret.
            authorize_url: The OAuth2 authorization URL.
            token_url: The OAuth2 token URL.
            userinfo_url: The OAuth2 user info URL.
            scope: The OAuth2 scope.
            redirect_uri: The OAuth2 redirect URI.
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._authorize_url = authorize_url
        self._token_url = token_url
        self._userinfo_url = userinfo_url
        self._scope = scope
        self._redirect_uri = redirect_uri
        
        if not self._client_id or not self._client_secret:
            raise ValueError("Client ID and client secret are required for OAuth2 authentication")
        
        if not self._authorize_url or not self._token_url or not self._userinfo_url:
            raise ValueError("Authorization URL, token URL, and user info URL are required for OAuth2 authentication")
    
    @property
    def provider(self) -> AuthProvider:
        """
        Get the authentication provider.
        
        Returns:
            The authentication provider.
        """
        return AuthProvider.OAUTH2
    
    def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            credentials: The credentials to authenticate with.
            
        Returns:
            The authenticated user, or None if authentication failed.
        """
        # In a real implementation, this would validate the credentials against the OAuth2 provider
        # For now, we just return a dummy user
        if credentials.get("code"):
            return User(
                id="1",
                username="oauth2_user",
                email="oauth2_user@example.com",
                first_name="OAuth2",
                last_name="User",
                roles=["user"],
                permissions=["read"],
            )
        
        return None
    
    def generate_token(self, user: User) -> str:
        """
        Generate an authentication token for a user.
        
        Args:
            user: The user to generate a token for.
            
        Returns:
            The authentication token.
        """
        # In a real implementation, this would generate a token from the OAuth2 provider
        # For now, we just return a dummy token
        return "oauth2_token"
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate an authentication token.
        
        Args:
            token: The token to validate.
            
        Returns:
            The authenticated user, or None if the token is invalid.
        """
        # In a real implementation, this would validate the token against the OAuth2 provider
        # For now, we just return a dummy user
        if token == "oauth2_token":
            return User(
                id="1",
                username="oauth2_user",
                email="oauth2_user@example.com",
                first_name="OAuth2",
                last_name="User",
                roles=["user"],
                permissions=["read"],
            )
        
        return None
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Get the OAuth2 authorization URL.
        
        Args:
            state: The state parameter for the OAuth2 authorization request.
            
        Returns:
            The OAuth2 authorization URL.
        """
        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
        }
        
        if self._scope:
            params["scope"] = self._scope
        
        if state:
            params["state"] = state
        
        # Build the URL
        url = self._authorize_url
        
        if "?" in url:
            url += "&"
        else:
            url += "?"
        
        url += "&".join(f"{key}={value}" for key, value in params.items())
        
        return url
    
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange an authorization code for an access token.
        
        Args:
            code: The authorization code.
            
        Returns:
            The access token response.
        """
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
        }
        
        response = requests.post(self._token_url, data=data)
        response.raise_for_status()
        
        return response.json()
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get user information from the OAuth2 provider.
        
        Args:
            access_token: The access token.
            
        Returns:
            The user information.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        
        response = requests.get(self._userinfo_url, headers=headers)
        response.raise_for_status()
        
        return response.json()


class AuthManager:
    """
    Authentication manager.
    
    This class is responsible for managing authentication backends and authenticating users.
    """
    
    def __init__(self):
        """
        Initialize the authentication manager.
        """
        self._backends: Dict[AuthProvider, AuthBackend] = {}
        self._default_backend: Optional[AuthBackend] = None
        self._lock = threading.RLock()
    
    def register_backend(self, backend: AuthBackend, default: bool = False) -> None:
        """
        Register an authentication backend.
        
        Args:
            backend: The authentication backend to register.
            default: Whether to set this backend as the default.
        """
        with self._lock:
            self._backends[backend.provider] = backend
            
            if default or not self._default_backend:
                self._default_backend = backend
    
    def get_backend(self, provider: Optional[AuthProvider] = None) -> Optional[AuthBackend]:
        """
        Get an authentication backend.
        
        Args:
            provider: The authentication provider. If None, returns the default backend.
            
        Returns:
            The authentication backend, or None if no backend is found.
        """
        with self._lock:
            if provider:
                return self._backends.get(provider)
            
            return self._default_backend
    
    def authenticate(
        self,
        credentials: Dict[str, Any],
        provider: Optional[AuthProvider] = None,
    ) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            credentials: The credentials to authenticate with.
            provider: The authentication provider. If None, uses the default backend.
            
        Returns:
            The authenticated user, or None if authentication failed.
        """
        backend = self.get_backend(provider)
        
        if not backend:
            log.error(f"No authentication backend found for provider {provider}")
            return None
        
        return backend.authenticate(credentials)
    
    def generate_token(
        self,
        user: User,
        provider: Optional[AuthProvider] = None,
    ) -> str:
        """
        Generate an authentication token for a user.
        
        Args:
            user: The user to generate a token for.
            provider: The authentication provider. If None, uses the default backend.
            
        Returns:
            The authentication token.
        """
        backend = self.get_backend(provider)
        
        if not backend:
            log.error(f"No authentication backend found for provider {provider}")
            raise ValueError(f"No authentication backend found for provider {provider}")
        
        return backend.generate_token(user)
    
    def validate_token(
        self,
        token: str,
        provider: Optional[AuthProvider] = None,
    ) -> Optional[User]:
        """
        Validate an authentication token.
        
        Args:
            token: The token to validate.
            provider: The authentication provider. If None, uses the default backend.
            
        Returns:
            The authenticated user, or None if the token is invalid.
        """
        backend = self.get_backend(provider)
        
        if not backend:
            log.error(f"No authentication backend found for provider {provider}")
            return None
        
        return backend.validate_token(token)


# Global authentication manager instance
_auth_manager = None
_auth_manager_lock = threading.RLock()


def get_auth_manager() -> AuthManager:
    """
    Get the global authentication manager instance.
    
    Returns:
        The global authentication manager instance.
    """
    global _auth_manager
    
    with _auth_manager_lock:
        if _auth_manager is None:
            _auth_manager = AuthManager()
            
            # Register the default JWT authentication backend
            config = get_config()
            
            if config.secret_key:
                jwt_backend = JWTAuthBackend(
                    secret_key=config.secret_key,
                    algorithm=config.jwt_algorithm,
                    expiration=config.jwt_expiration,
                )
                
                _auth_manager.register_backend(jwt_backend, default=True)
        
        return _auth_manager
