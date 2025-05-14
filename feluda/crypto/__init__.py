"""
Cryptography Package

This package provides advanced cryptographic capabilities for Feluda.
"""

from feluda.crypto.homomorphic import (
    HomomorphicEncryptionBackend,
    PyfhelBackend,
    TenSEALBackend,
)
from feluda.crypto.secure_multiparty import (
    PySyftBackend,
    SecureMultiPartyComputationBackend,
    TFEncryptedBackend,
)
from feluda.crypto.zero_knowledge import (
    CircomSnarkJSBackend,
    ZeroKnowledgeProofBackend,
)
from feluda.crypto.zkp import (
    PedersenCommitment,
    PedersenCommitter,
    SchnorrProver,
    SchnorrVerifier,
    ZKPManager,
    ZKProof,
    ZKProver,
    ZKRangeProof,
    ZKRangeProver,
    ZKRangeVerifier,
    ZKVerifier,
    get_zkp_manager,
)

__all__ = [
    # Homomorphic encryption
    "HomomorphicEncryptionBackend",
    "PyfhelBackend",
    "TenSEALBackend",

    # Zero-knowledge proofs
    "ZeroKnowledgeProofBackend",
    "CircomSnarkJSBackend",

    # Secure multi-party computation
    "SecureMultiPartyComputationBackend",
    "PySyftBackend",
    "TFEncryptedBackend",

    # Zero-knowledge proofs (ZKP)
    "PedersenCommitment",
    "PedersenCommitter",
    "SchnorrProver",
    "SchnorrVerifier",
    "ZKPManager",
    "ZKProof",
    "ZKProver",
    "ZKRangeProof",
    "ZKRangeProver",
    "ZKRangeVerifier",
    "ZKVerifier",
    "get_zkp_manager",
]
