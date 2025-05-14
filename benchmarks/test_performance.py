"""
Performance benchmarks for Feluda.

This module contains benchmarks for various components of Feluda.
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

from feluda.performance.numba_optimizations import optional_njit
from feluda.verification.vector_operations import (
    vector_add,
    vector_dot_product,
    vector_magnitude,
    vector_normalize,
)


# Define some test data
@pytest.fixture
def small_vectors():
    """Small vectors for benchmarking."""
    return np.random.random(100), np.random.random(100)


@pytest.fixture
def medium_vectors():
    """Medium vectors for benchmarking."""
    return np.random.random(1000), np.random.random(1000)


@pytest.fixture
def large_vectors():
    """Large vectors for benchmarking."""
    return np.random.random(10000), np.random.random(10000)


# Define some functions to benchmark
def python_vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two vectors using pure Python."""
    return [a[i] + b[i] for i in range(len(a))]


def numpy_vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two vectors using NumPy."""
    return a + b


@optional_njit
def numba_vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two vectors using Numba."""
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result


def python_vector_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two vectors using pure Python."""
    return sum(a[i] * b[i] for i in range(len(a)))


def numpy_vector_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two vectors using NumPy."""
    return np.dot(a, b)


@optional_njit
def numba_vector_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two vectors using Numba."""
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


# Benchmarks for vector addition
def test_python_vector_add_small(benchmark, small_vectors):
    """Benchmark Python vector addition with small vectors."""
    a, b = small_vectors
    benchmark(python_vector_add, a, b)


def test_numpy_vector_add_small(benchmark, small_vectors):
    """Benchmark NumPy vector addition with small vectors."""
    a, b = small_vectors
    benchmark(numpy_vector_add, a, b)


def test_numba_vector_add_small(benchmark, small_vectors):
    """Benchmark Numba vector addition with small vectors."""
    a, b = small_vectors
    benchmark(numba_vector_add, a, b)


def test_feluda_vector_add_small(benchmark, small_vectors):
    """Benchmark Feluda vector addition with small vectors."""
    a, b = small_vectors
    benchmark(vector_add, a, b)


def test_python_vector_add_medium(benchmark, medium_vectors):
    """Benchmark Python vector addition with medium vectors."""
    a, b = medium_vectors
    benchmark(python_vector_add, a, b)


def test_numpy_vector_add_medium(benchmark, medium_vectors):
    """Benchmark NumPy vector addition with medium vectors."""
    a, b = medium_vectors
    benchmark(numpy_vector_add, a, b)


def test_numba_vector_add_medium(benchmark, medium_vectors):
    """Benchmark Numba vector addition with medium vectors."""
    a, b = medium_vectors
    benchmark(numba_vector_add, a, b)


def test_feluda_vector_add_medium(benchmark, medium_vectors):
    """Benchmark Feluda vector addition with medium vectors."""
    a, b = medium_vectors
    benchmark(vector_add, a, b)


def test_python_vector_add_large(benchmark, large_vectors):
    """Benchmark Python vector addition with large vectors."""
    a, b = large_vectors
    benchmark(python_vector_add, a, b)


def test_numpy_vector_add_large(benchmark, large_vectors):
    """Benchmark NumPy vector addition with large vectors."""
    a, b = large_vectors
    benchmark(numpy_vector_add, a, b)


def test_numba_vector_add_large(benchmark, large_vectors):
    """Benchmark Numba vector addition with large vectors."""
    a, b = large_vectors
    benchmark(numba_vector_add, a, b)


def test_feluda_vector_add_large(benchmark, large_vectors):
    """Benchmark Feluda vector addition with large vectors."""
    a, b = large_vectors
    benchmark(vector_add, a, b)


# Benchmarks for vector dot product
def test_python_vector_dot_product_small(benchmark, small_vectors):
    """Benchmark Python vector dot product with small vectors."""
    a, b = small_vectors
    benchmark(python_vector_dot_product, a, b)


def test_numpy_vector_dot_product_small(benchmark, small_vectors):
    """Benchmark NumPy vector dot product with small vectors."""
    a, b = small_vectors
    benchmark(numpy_vector_dot_product, a, b)


def test_numba_vector_dot_product_small(benchmark, small_vectors):
    """Benchmark Numba vector dot product with small vectors."""
    a, b = small_vectors
    benchmark(numba_vector_dot_product, a, b)


def test_feluda_vector_dot_product_small(benchmark, small_vectors):
    """Benchmark Feluda vector dot product with small vectors."""
    a, b = small_vectors
    benchmark(vector_dot_product, a, b)


def test_python_vector_dot_product_medium(benchmark, medium_vectors):
    """Benchmark Python vector dot product with medium vectors."""
    a, b = medium_vectors
    benchmark(python_vector_dot_product, a, b)


def test_numpy_vector_dot_product_medium(benchmark, medium_vectors):
    """Benchmark NumPy vector dot product with medium vectors."""
    a, b = medium_vectors
    benchmark(numpy_vector_dot_product, a, b)


def test_numba_vector_dot_product_medium(benchmark, medium_vectors):
    """Benchmark Numba vector dot product with medium vectors."""
    a, b = medium_vectors
    benchmark(numba_vector_dot_product, a, b)


def test_feluda_vector_dot_product_medium(benchmark, medium_vectors):
    """Benchmark Feluda vector dot product with medium vectors."""
    a, b = medium_vectors
    benchmark(vector_dot_product, a, b)


def test_python_vector_dot_product_large(benchmark, large_vectors):
    """Benchmark Python vector dot product with large vectors."""
    a, b = large_vectors
    benchmark(python_vector_dot_product, a, b)


def test_numpy_vector_dot_product_large(benchmark, large_vectors):
    """Benchmark NumPy vector dot product with large vectors."""
    a, b = large_vectors
    benchmark(numpy_vector_dot_product, a, b)


def test_numba_vector_dot_product_large(benchmark, large_vectors):
    """Benchmark Numba vector dot product with large vectors."""
    a, b = large_vectors
    benchmark(numba_vector_dot_product, a, b)


def test_feluda_vector_dot_product_large(benchmark, large_vectors):
    """Benchmark Feluda vector dot product with large vectors."""
    a, b = large_vectors
    benchmark(vector_dot_product, a, b)


# Benchmarks for vector magnitude
def test_vector_magnitude_small(benchmark, small_vectors):
    """Benchmark vector magnitude with small vectors."""
    a, _ = small_vectors
    benchmark(vector_magnitude, a)


def test_vector_magnitude_medium(benchmark, medium_vectors):
    """Benchmark vector magnitude with medium vectors."""
    a, _ = medium_vectors
    benchmark(vector_magnitude, a)


def test_vector_magnitude_large(benchmark, large_vectors):
    """Benchmark vector magnitude with large vectors."""
    a, _ = large_vectors
    benchmark(vector_magnitude, a)


# Benchmarks for vector normalization
def test_vector_normalize_small(benchmark, small_vectors):
    """Benchmark vector normalization with small vectors."""
    a, _ = small_vectors
    benchmark(vector_normalize, a)


def test_vector_normalize_medium(benchmark, medium_vectors):
    """Benchmark vector normalization with medium vectors."""
    a, _ = medium_vectors
    benchmark(vector_normalize, a)


def test_vector_normalize_large(benchmark, large_vectors):
    """Benchmark vector normalization with large vectors."""
    a, _ = large_vectors
    benchmark(vector_normalize, a)
