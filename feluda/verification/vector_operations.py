"""
Vector Operations Module

This module provides pure functions for vector operations with formal verification hooks.
These functions are designed to be verified using tools like CrossHair.
"""

import math
from typing import List, Tuple, Union

import deal
import numpy as np


@deal.pre(lambda v: isinstance(v, (list, np.ndarray)))
@deal.pre(lambda v: len(v) > 0)
@deal.post(lambda result: isinstance(result, float))
@deal.post(lambda result: result >= 0)
@deal.ensure(lambda v, result: math.isclose(result, math.sqrt(sum(x * x for x in v)), abs_tol=1e-10))
def vector_norm(v: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the Euclidean (L2) norm of a vector.
    
    This function is designed to be formally verified.
    
    Args:
        v: The input vector.
        
    Returns:
        The Euclidean norm of the vector.
        
    Raises:
        ValueError: If the input is not a valid vector.
    """
    if isinstance(v, np.ndarray):
        return float(np.linalg.norm(v))
    return math.sqrt(sum(x * x for x in v))


@deal.pre(lambda v: isinstance(v, (list, np.ndarray)))
@deal.pre(lambda v: len(v) > 0)
@deal.post(lambda result: isinstance(result, type(v)))
@deal.post(lambda result: len(result) == len(v))
@deal.ensure(lambda v, result: all(math.isclose(vector_norm(result), 1.0, abs_tol=1e-10) for _ in [0]))
def normalize_vector(v: Union[List[float], np.ndarray]) -> Union[List[float], np.ndarray]:
    """
    Normalize a vector to unit length.
    
    This function is designed to be formally verified.
    
    Args:
        v: The input vector.
        
    Returns:
        The normalized vector.
        
    Raises:
        ValueError: If the input is not a valid vector or has zero norm.
    """
    norm = vector_norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    
    if isinstance(v, np.ndarray):
        return v / norm
    return [x / norm for x in v]


@deal.pre(lambda v1, v2: isinstance(v1, (list, np.ndarray)) and isinstance(v2, (list, np.ndarray)))
@deal.pre(lambda v1, v2: len(v1) > 0 and len(v2) > 0)
@deal.pre(lambda v1, v2: len(v1) == len(v2))
@deal.post(lambda result: isinstance(result, float))
@deal.post(lambda result: -1.0 <= result <= 1.0)
@deal.ensure(lambda v1, v2, result: math.isclose(result, sum(a * b for a, b in zip(normalize_vector(v1), normalize_vector(v2))), abs_tol=1e-10))
def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    This function is designed to be formally verified.
    
    Args:
        v1: The first vector.
        v2: The second vector.
        
    Returns:
        The cosine similarity between the vectors.
        
    Raises:
        ValueError: If the inputs are not valid vectors or have different dimensions.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute cosine similarity with zero vectors")
    
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    return dot_product / (norm1 * norm2)


@deal.pre(lambda v1, v2: isinstance(v1, (list, np.ndarray)) and isinstance(v2, (list, np.ndarray)))
@deal.pre(lambda v1, v2: len(v1) > 0 and len(v2) > 0)
@deal.pre(lambda v1, v2: len(v1) == len(v2))
@deal.post(lambda result: isinstance(result, float))
@deal.post(lambda result: result >= 0)
@deal.ensure(lambda v1, v2, result: math.isclose(result, math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2))), abs_tol=1e-10))
def euclidean_distance(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the Euclidean distance between two vectors.
    
    This function is designed to be formally verified.
    
    Args:
        v1: The first vector.
        v2: The second vector.
        
    Returns:
        The Euclidean distance between the vectors.
        
    Raises:
        ValueError: If the inputs are not valid vectors or have different dimensions.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return float(np.linalg.norm(v1 - v2))
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


@deal.pre(lambda vectors: isinstance(vectors, (list, np.ndarray)))
@deal.pre(lambda vectors: len(vectors) > 0)
@deal.pre(lambda vectors: all(isinstance(v, (list, np.ndarray)) for v in vectors))
@deal.pre(lambda vectors: all(len(v) > 0 for v in vectors))
@deal.pre(lambda vectors: all(len(v) == len(vectors[0]) for v in vectors))
@deal.post(lambda result: isinstance(result, (list, np.ndarray)))
@deal.post(lambda result: len(result) == len(vectors[0]))
def mean_vector(vectors: Union[List[List[float]], List[np.ndarray], np.ndarray]) -> Union[List[float], np.ndarray]:
    """
    Calculate the mean vector from a list of vectors.
    
    This function is designed to be formally verified.
    
    Args:
        vectors: A list of vectors.
        
    Returns:
        The mean vector.
        
    Raises:
        ValueError: If the input is not a valid list of vectors.
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute mean of empty list")
    
    if isinstance(vectors, np.ndarray):
        return np.mean(vectors, axis=0)
    
    if isinstance(vectors[0], np.ndarray):
        return np.mean(np.array(vectors), axis=0)
    
    n = len(vectors)
    dim = len(vectors[0])
    result = [0.0] * dim
    
    for v in vectors:
        for i in range(dim):
            result[i] += v[i]
    
    return [x / n for x in result]


@deal.pre(lambda v1, v2, t: isinstance(v1, (list, np.ndarray)) and isinstance(v2, (list, np.ndarray)))
@deal.pre(lambda v1, v2, t: len(v1) > 0 and len(v2) > 0)
@deal.pre(lambda v1, v2, t: len(v1) == len(v2))
@deal.pre(lambda v1, v2, t: 0 <= t <= 1)
@deal.post(lambda result: isinstance(result, (list, np.ndarray)))
@deal.post(lambda result, v1, v2: len(result) == len(v1))
@deal.ensure(lambda v1, v2, t, result: all(math.isclose(result[i], (1 - t) * v1[i] + t * v2[i], abs_tol=1e-10) for i in range(len(v1))))
def interpolate_vectors(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray], t: float) -> Union[List[float], np.ndarray]:
    """
    Interpolate between two vectors.
    
    This function is designed to be formally verified.
    
    Args:
        v1: The first vector.
        v2: The second vector.
        t: The interpolation parameter (0 <= t <= 1).
        
    Returns:
        The interpolated vector.
        
    Raises:
        ValueError: If the inputs are not valid vectors or have different dimensions.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    
    if not 0 <= t <= 1:
        raise ValueError("Interpolation parameter must be between 0 and 1")
    
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return (1 - t) * v1 + t * v2
    
    return [(1 - t) * a + t * b for a, b in zip(v1, v2)]
