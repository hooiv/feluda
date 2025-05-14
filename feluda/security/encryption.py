"""
Encryption module for Feluda.

This module provides encryption features for Feluda.
"""

import abc
import base64
import enum
import hashlib
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class EncryptionProvider(str, enum.Enum):
    """Enum for encryption providers."""
    
    AES = "aes"
    RSA = "rsa"
    FERNET = "fernet"
    CUSTOM = "custom"


class EncryptionBackend(abc.ABC):
    """
    Base class for encryption backends.
    
    This class defines the interface for encryption backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @property
    @abc.abstractmethod
    def provider(self) -> EncryptionProvider:
        """
        Get the encryption provider.
        
        Returns:
            The encryption provider.
        """
        pass
    
    @abc.abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: The data to encrypt.
            
        Returns:
            The encrypted data.
        """
        pass
    
    @abc.abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            data: The data to decrypt.
            
        Returns:
            The decrypted data.
        """
        pass


class AESEncryptionBackend(EncryptionBackend):
    """
    AES encryption backend.
    
    This class implements encryption using AES.
    """
    
    def __init__(self, key: Optional[bytes] = None, iv: Optional[bytes] = None):
        """
        Initialize the AES encryption backend.
        
        Args:
            key: The encryption key. If None, a key is derived from the secret key.
            iv: The initialization vector. If None, a random IV is generated.
        """
        config = get_config()
        
        if key is None:
            # Derive a key from the secret key
            if not config.secret_key:
                raise ValueError("Secret key is required for AES encryption")
            
            # Use PBKDF2 to derive a key
            salt = b"feluda-aes"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                iterations=100000,
                backend=default_backend(),
            )
            
            self._key = kdf.derive(config.secret_key.encode())
        else:
            self._key = key
        
        if iv is None:
            # Generate a random IV
            self._iv = os.urandom(16)
        else:
            self._iv = iv
    
    @property
    def provider(self) -> EncryptionProvider:
        """
        Get the encryption provider.
        
        Returns:
            The encryption provider.
        """
        return EncryptionProvider.AES
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using AES.
        
        Args:
            data: The data to encrypt.
            
        Returns:
            The encrypted data.
        """
        # Pad the data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Create the cipher
        cipher = Cipher(
            algorithms.AES(self._key),
            modes.CBC(self._iv),
            backend=default_backend(),
        )
        
        # Encrypt the data
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Prepend the IV to the encrypted data
        return self._iv + encrypted_data
    
    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data using AES.
        
        Args:
            data: The data to decrypt.
            
        Returns:
            The decrypted data.
        """
        # Extract the IV from the data
        iv = data[:16]
        encrypted_data = data[16:]
        
        # Create the cipher
        cipher = Cipher(
            algorithms.AES(self._key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        
        # Decrypt the data
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Unpad the data
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()


class RSAEncryptionBackend(EncryptionBackend):
    """
    RSA encryption backend.
    
    This class implements encryption using RSA.
    """
    
    def __init__(
        self,
        public_key: Optional[bytes] = None,
        private_key: Optional[bytes] = None,
        password: Optional[bytes] = None,
    ):
        """
        Initialize the RSA encryption backend.
        
        Args:
            public_key: The public key in PEM format.
            private_key: The private key in PEM format.
            password: The password for the private key.
        """
        if public_key is None and private_key is None:
            # Generate a new key pair
            private_key_obj = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend(),
            )
            
            self._private_key = private_key_obj
            self._public_key = private_key_obj.public_key()
        else:
            # Load the keys
            if private_key is not None:
                self._private_key = load_pem_private_key(
                    private_key,
                    password=password,
                    backend=default_backend(),
                )
            else:
                self._private_key = None
            
            if public_key is not None:
                self._public_key = load_pem_public_key(
                    public_key,
                    backend=default_backend(),
                )
            else:
                self._public_key = None
    
    @property
    def provider(self) -> EncryptionProvider:
        """
        Get the encryption provider.
        
        Returns:
            The encryption provider.
        """
        return EncryptionProvider.RSA
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using RSA.
        
        Args:
            data: The data to encrypt.
            
        Returns:
            The encrypted data.
        """
        if self._public_key is None:
            raise ValueError("Public key is required for RSA encryption")
        
        # RSA can only encrypt small amounts of data, so we use a hybrid approach
        # Generate a random AES key
        aes_key = os.urandom(32)
        
        # Encrypt the data with AES
        aes_backend = AESEncryptionBackend(key=aes_key)
        encrypted_data = aes_backend.encrypt(data)
        
        # Encrypt the AES key with RSA
        encrypted_key = self._public_key.encrypt(
            aes_key,
            asymmetric_padding.OAEP(
                mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        
        # Prepend the encrypted key length and the encrypted key to the encrypted data
        key_length = len(encrypted_key).to_bytes(4, byteorder="big")
        return key_length + encrypted_key + encrypted_data
    
    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data using RSA.
        
        Args:
            data: The data to decrypt.
            
        Returns:
            The decrypted data.
        """
        if self._private_key is None:
            raise ValueError("Private key is required for RSA decryption")
        
        # Extract the encrypted key length, the encrypted key, and the encrypted data
        key_length = int.from_bytes(data[:4], byteorder="big")
        encrypted_key = data[4:4 + key_length]
        encrypted_data = data[4 + key_length:]
        
        # Decrypt the AES key with RSA
        aes_key = self._private_key.decrypt(
            encrypted_key,
            asymmetric_padding.OAEP(
                mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        
        # Decrypt the data with AES
        aes_backend = AESEncryptionBackend(key=aes_key)
        return aes_backend.decrypt(encrypted_data)


class FernetEncryptionBackend(EncryptionBackend):
    """
    Fernet encryption backend.
    
    This class implements encryption using Fernet.
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize the Fernet encryption backend.
        
        Args:
            key: The encryption key. If None, a key is derived from the secret key.
        """
        config = get_config()
        
        if key is None:
            # Derive a key from the secret key
            if not config.secret_key:
                raise ValueError("Secret key is required for Fernet encryption")
            
            # Use PBKDF2 to derive a key
            salt = b"feluda-fernet"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                iterations=100000,
                backend=default_backend(),
            )
            
            key_bytes = kdf.derive(config.secret_key.encode())
            
            # Encode the key in base64
            self._key = base64.urlsafe_b64encode(key_bytes)
        else:
            self._key = key
        
        self._fernet = Fernet(self._key)
    
    @property
    def provider(self) -> EncryptionProvider:
        """
        Get the encryption provider.
        
        Returns:
            The encryption provider.
        """
        return EncryptionProvider.FERNET
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using Fernet.
        
        Args:
            data: The data to encrypt.
            
        Returns:
            The encrypted data.
        """
        return self._fernet.encrypt(data)
    
    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data using Fernet.
        
        Args:
            data: The data to decrypt.
            
        Returns:
            The decrypted data.
        """
        return self._fernet.decrypt(data)


class EncryptionManager:
    """
    Encryption manager.
    
    This class is responsible for managing encryption backends and encrypting/decrypting data.
    """
    
    def __init__(self):
        """
        Initialize the encryption manager.
        """
        self._backends: Dict[EncryptionProvider, EncryptionBackend] = {}
        self._default_backend: Optional[EncryptionBackend] = None
        self._lock = threading.RLock()
    
    def register_backend(self, backend: EncryptionBackend, default: bool = False) -> None:
        """
        Register an encryption backend.
        
        Args:
            backend: The encryption backend to register.
            default: Whether to set this backend as the default.
        """
        with self._lock:
            self._backends[backend.provider] = backend
            
            if default or not self._default_backend:
                self._default_backend = backend
    
    def get_backend(self, provider: Optional[EncryptionProvider] = None) -> Optional[EncryptionBackend]:
        """
        Get an encryption backend.
        
        Args:
            provider: The encryption provider. If None, returns the default backend.
            
        Returns:
            The encryption backend, or None if no backend is found.
        """
        with self._lock:
            if provider:
                return self._backends.get(provider)
            
            return self._default_backend
    
    def encrypt(
        self,
        data: Union[str, bytes],
        provider: Optional[EncryptionProvider] = None,
    ) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: The data to encrypt.
            provider: The encryption provider. If None, uses the default backend.
            
        Returns:
            The encrypted data.
        """
        backend = self.get_backend(provider)
        
        if not backend:
            log.error(f"No encryption backend found for provider {provider}")
            raise ValueError(f"No encryption backend found for provider {provider}")
        
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode()
        
        return backend.encrypt(data)
    
    def decrypt(
        self,
        data: bytes,
        provider: Optional[EncryptionProvider] = None,
    ) -> bytes:
        """
        Decrypt data.
        
        Args:
            data: The data to decrypt.
            provider: The encryption provider. If None, uses the default backend.
            
        Returns:
            The decrypted data.
        """
        backend = self.get_backend(provider)
        
        if not backend:
            log.error(f"No encryption backend found for provider {provider}")
            raise ValueError(f"No encryption backend found for provider {provider}")
        
        return backend.decrypt(data)
    
    def encrypt_to_base64(
        self,
        data: Union[str, bytes],
        provider: Optional[EncryptionProvider] = None,
    ) -> str:
        """
        Encrypt data and encode it in base64.
        
        Args:
            data: The data to encrypt.
            provider: The encryption provider. If None, uses the default backend.
            
        Returns:
            The encrypted data encoded in base64.
        """
        encrypted_data = self.encrypt(data, provider)
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_from_base64(
        self,
        data: str,
        provider: Optional[EncryptionProvider] = None,
    ) -> bytes:
        """
        Decrypt data from base64.
        
        Args:
            data: The encrypted data encoded in base64.
            provider: The encryption provider. If None, uses the default backend.
            
        Returns:
            The decrypted data.
        """
        encrypted_data = base64.b64decode(data)
        return self.decrypt(encrypted_data, provider)


# Global encryption manager instance
_encryption_manager = None
_encryption_manager_lock = threading.RLock()


def get_encryption_manager() -> EncryptionManager:
    """
    Get the global encryption manager instance.
    
    Returns:
        The global encryption manager instance.
    """
    global _encryption_manager
    
    with _encryption_manager_lock:
        if _encryption_manager is None:
            _encryption_manager = EncryptionManager()
            
            # Register the default Fernet encryption backend
            fernet_backend = FernetEncryptionBackend()
            _encryption_manager.register_backend(fernet_backend, default=True)
            
            # Register the AES encryption backend
            aes_backend = AESEncryptionBackend()
            _encryption_manager.register_backend(aes_backend)
        
        return _encryption_manager
