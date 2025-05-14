"""
Performance Package

This package provides performance optimization tools for Feluda.
"""

from feluda.performance.hardware_acceleration import (
    HardwareInfo,
    HardwareProfile,
    HardwareType,
    HardwareVendor,
    get_hardware_profile,
)
from feluda.performance.numba_optimizations import (
    cosine_similarity_numba,
    euclidean_distance_numba,
    is_numba_available,
    matrix_matrix_multiply_numba,
    matrix_vector_multiply_numba,
    normalize_vectors_numba,
    optional_jit,
    optional_njit,
    pairwise_distances_numba,
)

__all__ = [
    # Hardware acceleration
    "HardwareType",
    "HardwareVendor",
    "HardwareInfo",
    "HardwareProfile",
    "get_hardware_profile",
    
    # Numba optimizations
    "is_numba_available",
    "optional_jit",
    "optional_njit",
    "cosine_similarity_numba",
    "euclidean_distance_numba",
    "pairwise_distances_numba",
    "normalize_vectors_numba",
    "matrix_vector_multiply_numba",
    "matrix_matrix_multiply_numba",
]
