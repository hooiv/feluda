"""
Cryptography benchmarks for Feluda.

This module contains benchmarks for the cryptography components of Feluda.
Run with: pytest benchmarks/ --benchmark-only
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from feluda.crypto.homomorphic import (
        HomomorphicBackend,
        HomomorphicConfig,
        HomomorphicInterface,
        PyfhelBackend,
    )
    HOMOMORPHIC_AVAILABLE = True
except ImportError:
    HOMOMORPHIC_AVAILABLE = False

try:
    from feluda.crypto.zero_knowledge import (
        ZeroKnowledgeBackend,
        ZeroKnowledgeConfig,
        ZeroKnowledgeInterface,
        CircomSnarkJSBackend,
    )
    ZERO_KNOWLEDGE_AVAILABLE = True
except ImportError:
    ZERO_KNOWLEDGE_AVAILABLE = False

try:
    from feluda.crypto.secure_multiparty import (
        SecureMultipartyBackend,
        SecureMultipartyConfig,
        SecureMultipartyInterface,
        PySyftBackend,
    )
    SECURE_MULTIPARTY_AVAILABLE = True
except ImportError:
    SECURE_MULTIPARTY_AVAILABLE = False


# Define some fixtures
@pytest.fixture
def homomorphic_config():
    """Create a homomorphic encryption configuration."""
    if not HOMOMORPHIC_AVAILABLE:
        pytest.skip("Homomorphic encryption not available")
    
    return HomomorphicConfig(
        backend=HomomorphicBackend.PYFHEL,
        key_size=2048,
        security_level=128,
    )


@pytest.fixture
def zero_knowledge_config():
    """Create a zero-knowledge proof configuration."""
    if not ZERO_KNOWLEDGE_AVAILABLE:
        pytest.skip("Zero-knowledge proofs not available")
    
    return ZeroKnowledgeConfig(
        backend=ZeroKnowledgeBackend.CIRCOM_SNARKJS,
        curve="bn128",
    )


@pytest.fixture
def secure_multiparty_config():
    """Create a secure multi-party computation configuration."""
    if not SECURE_MULTIPARTY_AVAILABLE:
        pytest.skip("Secure multi-party computation not available")
    
    return SecureMultipartyConfig(
        backend=SecureMultipartyBackend.PYSYFT,
        num_parties=3,
    )


# Benchmarks for homomorphic encryption
@pytest.mark.skipif(not HOMOMORPHIC_AVAILABLE, reason="Homomorphic encryption not available")
def test_homomorphic_config_to_dict(benchmark, homomorphic_config):
    """Benchmark homomorphic config to_dict."""
    benchmark(homomorphic_config.to_dict)


@pytest.mark.skipif(not HOMOMORPHIC_AVAILABLE, reason="Homomorphic encryption not available")
def test_homomorphic_config_from_dict(benchmark, homomorphic_config):
    """Benchmark homomorphic config from_dict."""
    config_dict = homomorphic_config.to_dict()
    benchmark(HomomorphicConfig.from_dict, config_dict)


@pytest.mark.skipif(not HOMOMORPHIC_AVAILABLE, reason="Homomorphic encryption not available")
def test_homomorphic_interface_creation(benchmark, homomorphic_config):
    """Benchmark homomorphic interface creation."""
    benchmark(HomomorphicInterface, homomorphic_config)


@pytest.mark.skipif(not HOMOMORPHIC_AVAILABLE, reason="Homomorphic encryption not available")
def test_pyfhel_backend_creation(benchmark):
    """Benchmark Pyfhel backend creation."""
    benchmark(PyfhelBackend)


@pytest.mark.skipif(not HOMOMORPHIC_AVAILABLE, reason="Homomorphic encryption not available")
def test_pyfhel_backend_generate_keys(benchmark):
    """Benchmark Pyfhel backend key generation."""
    backend = PyfhelBackend()
    benchmark(backend.generate_keys)


# Benchmarks for zero-knowledge proofs
@pytest.mark.skipif(not ZERO_KNOWLEDGE_AVAILABLE, reason="Zero-knowledge proofs not available")
def test_zero_knowledge_config_to_dict(benchmark, zero_knowledge_config):
    """Benchmark zero-knowledge config to_dict."""
    benchmark(zero_knowledge_config.to_dict)


@pytest.mark.skipif(not ZERO_KNOWLEDGE_AVAILABLE, reason="Zero-knowledge proofs not available")
def test_zero_knowledge_config_from_dict(benchmark, zero_knowledge_config):
    """Benchmark zero-knowledge config from_dict."""
    config_dict = zero_knowledge_config.to_dict()
    benchmark(ZeroKnowledgeConfig.from_dict, config_dict)


@pytest.mark.skipif(not ZERO_KNOWLEDGE_AVAILABLE, reason="Zero-knowledge proofs not available")
def test_zero_knowledge_interface_creation(benchmark, zero_knowledge_config):
    """Benchmark zero-knowledge interface creation."""
    benchmark(ZeroKnowledgeInterface, zero_knowledge_config)


@pytest.mark.skipif(not ZERO_KNOWLEDGE_AVAILABLE, reason="Zero-knowledge proofs not available")
def test_circom_snarkjs_backend_creation(benchmark):
    """Benchmark CircomSnarkJS backend creation."""
    benchmark(CircomSnarkJSBackend)


# Benchmarks for secure multi-party computation
@pytest.mark.skipif(not SECURE_MULTIPARTY_AVAILABLE, reason="Secure multi-party computation not available")
def test_secure_multiparty_config_to_dict(benchmark, secure_multiparty_config):
    """Benchmark secure multi-party config to_dict."""
    benchmark(secure_multiparty_config.to_dict)


@pytest.mark.skipif(not SECURE_MULTIPARTY_AVAILABLE, reason="Secure multi-party computation not available")
def test_secure_multiparty_config_from_dict(benchmark, secure_multiparty_config):
    """Benchmark secure multi-party config from_dict."""
    config_dict = secure_multiparty_config.to_dict()
    benchmark(SecureMultipartyConfig.from_dict, config_dict)


@pytest.mark.skipif(not SECURE_MULTIPARTY_AVAILABLE, reason="Secure multi-party computation not available")
def test_secure_multiparty_interface_creation(benchmark, secure_multiparty_config):
    """Benchmark secure multi-party interface creation."""
    benchmark(SecureMultipartyInterface, secure_multiparty_config)


@pytest.mark.skipif(not SECURE_MULTIPARTY_AVAILABLE, reason="Secure multi-party computation not available")
def test_pysyft_backend_creation(benchmark):
    """Benchmark PySyft backend creation."""
    benchmark(PySyftBackend)
