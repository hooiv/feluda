"""
Verification Package

This package provides tools for formal verification of code.
"""

from feluda.verification.vector_operations import (
    cosine_similarity,
    euclidean_distance,
    interpolate_vectors,
    mean_vector,
    normalize_vector,
    vector_norm,
)

__all__ = [
    "vector_norm",
    "normalize_vector",
    "cosine_similarity",
    "euclidean_distance",
    "mean_vector",
    "interpolate_vectors",
]
