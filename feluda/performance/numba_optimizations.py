"""
Numba Optimizations Module

This module provides performance optimizations using Numba JIT compilation.
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import numpy as np

# Check if Numba is available
try:
    import numba
    from numba import jit, njit, prange
    
    NUMBA_AVAILABLE = True
except ImportError:
    # Create dummy decorators if Numba is not available
    def jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        
        def decorator(func):
            return func
        
        return decorator
    
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        
        def decorator(func):
            return func
        
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    
    NUMBA_AVAILABLE = False

log = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def optional_jit(
    *args,
    **kwargs,
):
    """
    Decorator that applies Numba JIT compilation if Numba is available.
    
    This decorator is a wrapper around Numba's jit decorator that falls back to
    the original function if Numba is not available.
    
    Args:
        *args: Positional arguments to pass to Numba's jit decorator.
        **kwargs: Keyword arguments to pass to Numba's jit decorator.
        
    Returns:
        The decorated function, JIT-compiled if Numba is available.
    """
    if not NUMBA_AVAILABLE:
        log.warning("Numba is not available. Using unoptimized function.")
        
        def decorator(func):
            return func
        
        # Handle both @optional_jit and @optional_jit()
        if len(args) == 1 and callable(args[0]):
            return args[0]
        
        return decorator
    
    return jit(*args, **kwargs)


def optional_njit(
    *args,
    **kwargs,
):
    """
    Decorator that applies Numba no-Python JIT compilation if Numba is available.
    
    This decorator is a wrapper around Numba's njit decorator that falls back to
    the original function if Numba is not available.
    
    Args:
        *args: Positional arguments to pass to Numba's njit decorator.
        **kwargs: Keyword arguments to pass to Numba's njit decorator.
        
    Returns:
        The decorated function, JIT-compiled if Numba is available.
    """
    if not NUMBA_AVAILABLE:
        log.warning("Numba is not available. Using unoptimized function.")
        
        def decorator(func):
            return func
        
        # Handle both @optional_njit and @optional_njit()
        if len(args) == 1 and callable(args[0]):
            return args[0]
        
        return decorator
    
    return njit(*args, **kwargs)


def is_numba_available() -> bool:
    """
    Check if Numba is available.
    
    Returns:
        True if Numba is available, False otherwise.
    """
    return NUMBA_AVAILABLE


@optional_njit
def cosine_similarity_numba(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors using Numba.
    
    Args:
        a: The first vector.
        b: The second vector.
        
    Returns:
        The cosine similarity between the vectors.
    """
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    
    for i in range(len(a)):
        dot_product += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot_product / (np.sqrt(norm_a) * np.sqrt(norm_b))


@optional_njit
def euclidean_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two vectors using Numba.
    
    Args:
        a: The first vector.
        b: The second vector.
        
    Returns:
        The Euclidean distance between the vectors.
    """
    sum_squared_diff = 0.0
    
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_squared_diff += diff * diff
    
    return np.sqrt(sum_squared_diff)


@optional_njit(parallel=True)
def pairwise_distances_numba(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Calculate pairwise distances between vectors in a matrix using Numba.
    
    Args:
        X: The matrix of vectors.
        metric: The distance metric to use. Options: "euclidean", "cosine".
        
    Returns:
        A matrix of pairwise distances.
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    
    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            if metric == "euclidean":
                dist = euclidean_distance_numba(X[i], X[j])
            elif metric == "cosine":
                dist = 1.0 - cosine_similarity_numba(X[i], X[j])
            else:
                # This will raise an error in Numba mode
                raise ValueError(f"Unknown metric: {metric}")
            
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


@optional_njit
def normalize_vectors_numba(X: np.ndarray) -> np.ndarray:
    """
    Normalize vectors in a matrix using Numba.
    
    Args:
        X: The matrix of vectors.
        
    Returns:
        The normalized matrix.
    """
    n_samples, n_features = X.shape
    normalized = np.zeros_like(X)
    
    for i in range(n_samples):
        norm = 0.0
        for j in range(n_features):
            norm += X[i, j] * X[i, j]
        
        norm = np.sqrt(norm)
        
        if norm > 0.0:
            for j in range(n_features):
                normalized[i, j] = X[i, j] / norm
    
    return normalized


@optional_njit(parallel=True)
def matrix_vector_multiply_numba(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply a matrix by a vector using Numba.
    
    Args:
        A: The matrix.
        b: The vector.
        
    Returns:
        The result of the multiplication.
    """
    n_rows = A.shape[0]
    n_cols = A.shape[1]
    result = np.zeros(n_rows, dtype=A.dtype)
    
    for i in prange(n_rows):
        for j in range(n_cols):
            result[i] += A[i, j] * b[j]
    
    return result


@optional_njit(parallel=True)
def matrix_matrix_multiply_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices using Numba.
    
    Args:
        A: The first matrix.
        B: The second matrix.
        
    Returns:
        The result of the multiplication.
    """
    n_rows_A = A.shape[0]
    n_cols_A = A.shape[1]
    n_cols_B = B.shape[1]
    result = np.zeros((n_rows_A, n_cols_B), dtype=A.dtype)
    
    for i in prange(n_rows_A):
        for j in range(n_cols_B):
            for k in range(n_cols_A):
                result[i, j] += A[i, k] * B[k, j]
    
    return result
