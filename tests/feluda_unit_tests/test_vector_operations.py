"""
Tests for the vector operations module.
"""

import math
import unittest

import numpy as np
import pytest
from hypothesis import given, strategies as st

from feluda.verification.vector_operations import (
    cosine_similarity,
    euclidean_distance,
    interpolate_vectors,
    mean_vector,
    normalize_vector,
    vector_norm,
)


class TestVectorOperations(unittest.TestCase):
    """Tests for the vector operations module."""
    
    def test_vector_norm(self):
        """Test vector_norm function."""
        # Test with list
        self.assertAlmostEqual(vector_norm([3, 4]), 5.0)
        self.assertAlmostEqual(vector_norm([1, 1, 1, 1]), 2.0)
        
        # Test with numpy array
        self.assertAlmostEqual(vector_norm(np.array([3, 4])), 5.0)
        self.assertAlmostEqual(vector_norm(np.array([1, 1, 1, 1])), 2.0)
    
    def test_normalize_vector(self):
        """Test normalize_vector function."""
        # Test with list
        normalized = normalize_vector([3, 4])
        self.assertAlmostEqual(normalized[0], 0.6)
        self.assertAlmostEqual(normalized[1], 0.8)
        self.assertAlmostEqual(vector_norm(normalized), 1.0)
        
        # Test with numpy array
        normalized = normalize_vector(np.array([3, 4]))
        self.assertAlmostEqual(normalized[0], 0.6)
        self.assertAlmostEqual(normalized[1], 0.8)
        self.assertAlmostEqual(vector_norm(normalized), 1.0)
    
    def test_normalize_vector_zero(self):
        """Test normalize_vector function with zero vector."""
        with self.assertRaises(ValueError):
            normalize_vector([0, 0])
        
        with self.assertRaises(ValueError):
            normalize_vector(np.array([0, 0]))
    
    def test_cosine_similarity(self):
        """Test cosine_similarity function."""
        # Test with lists
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertAlmostEqual(cosine_similarity([1, 1], [1, 0]), 1 / math.sqrt(2))
        
        # Test with numpy arrays
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([1, 0])), 1.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([-1, 0])), -1.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1, 1]), np.array([1, 0])), 1 / math.sqrt(2))
    
    def test_cosine_similarity_different_dimensions(self):
        """Test cosine_similarity function with vectors of different dimensions."""
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [0, 1, 0])
        
        with self.assertRaises(ValueError):
            cosine_similarity(np.array([1, 0]), np.array([0, 1, 0]))
    
    def test_cosine_similarity_zero_vectors(self):
        """Test cosine_similarity function with zero vectors."""
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])
        
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [0, 0])
        
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [0, 0])
    
    def test_euclidean_distance(self):
        """Test euclidean_distance function."""
        # Test with lists
        self.assertAlmostEqual(euclidean_distance([1, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(euclidean_distance([1, 1], [4, 5]), 5.0)
        
        # Test with numpy arrays
        self.assertAlmostEqual(euclidean_distance(np.array([1, 0]), np.array([0, 0])), 1.0)
        self.assertAlmostEqual(euclidean_distance(np.array([1, 1]), np.array([4, 5])), 5.0)
    
    def test_euclidean_distance_different_dimensions(self):
        """Test euclidean_distance function with vectors of different dimensions."""
        with self.assertRaises(ValueError):
            euclidean_distance([1, 0], [0, 1, 0])
        
        with self.assertRaises(ValueError):
            euclidean_distance(np.array([1, 0]), np.array([0, 1, 0]))
    
    def test_mean_vector(self):
        """Test mean_vector function."""
        # Test with lists
        mean = mean_vector([[1, 0], [0, 1], [1, 1]])
        self.assertAlmostEqual(mean[0], 2/3)
        self.assertAlmostEqual(mean[1], 2/3)
        
        # Test with numpy arrays
        mean = mean_vector([np.array([1, 0]), np.array([0, 1]), np.array([1, 1])])
        self.assertAlmostEqual(mean[0], 2/3)
        self.assertAlmostEqual(mean[1], 2/3)
        
        # Test with numpy array of arrays
        mean = mean_vector(np.array([[1, 0], [0, 1], [1, 1]]))
        self.assertAlmostEqual(mean[0], 2/3)
        self.assertAlmostEqual(mean[1], 2/3)
    
    def test_mean_vector_empty(self):
        """Test mean_vector function with empty list."""
        with self.assertRaises(ValueError):
            mean_vector([])
    
    def test_mean_vector_different_dimensions(self):
        """Test mean_vector function with vectors of different dimensions."""
        with self.assertRaises(ValueError):
            mean_vector([[1, 0], [0, 1, 0]])
    
    def test_interpolate_vectors(self):
        """Test interpolate_vectors function."""
        # Test with lists
        interp = interpolate_vectors([0, 0], [10, 10], 0.3)
        self.assertAlmostEqual(interp[0], 3.0)
        self.assertAlmostEqual(interp[1], 3.0)
        
        # Test with numpy arrays
        interp = interpolate_vectors(np.array([0, 0]), np.array([10, 10]), 0.3)
        self.assertAlmostEqual(interp[0], 3.0)
        self.assertAlmostEqual(interp[1], 3.0)
    
    def test_interpolate_vectors_different_dimensions(self):
        """Test interpolate_vectors function with vectors of different dimensions."""
        with self.assertRaises(ValueError):
            interpolate_vectors([1, 0], [0, 1, 0], 0.5)
        
        with self.assertRaises(ValueError):
            interpolate_vectors(np.array([1, 0]), np.array([0, 1, 0]), 0.5)
    
    def test_interpolate_vectors_invalid_t(self):
        """Test interpolate_vectors function with invalid t."""
        with self.assertRaises(ValueError):
            interpolate_vectors([1, 0], [0, 1], -0.1)
        
        with self.assertRaises(ValueError):
            interpolate_vectors([1, 0], [0, 1], 1.1)


class TestVectorOperationsHypothesis:
    """Hypothesis tests for the vector operations module."""
    
    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=10))
    def test_vector_norm_properties(self, v):
        """Test properties of vector_norm function."""
        # Filter out NaN and Inf values
        v = [x for x in v if not math.isnan(x) and not math.isinf(x)]
        if not v:
            return
        
        # The norm should be non-negative
        assert vector_norm(v) >= 0
        
        # The norm of a scaled vector should be the scale times the norm
        scale = 2.0
        scaled_v = [scale * x for x in v]
        assert math.isclose(vector_norm(scaled_v), abs(scale) * vector_norm(v), rel_tol=1e-10)
    
    @given(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_normalize_vector_properties(self, v, epsilon):
        """Test properties of normalize_vector function."""
        # Filter out NaN and Inf values
        v = [x for x in v if not math.isnan(x) and not math.isinf(x)]
        if not v:
            return
        
        # Skip zero vectors
        if all(abs(x) < epsilon for x in v):
            return
        
        # The norm of a normalized vector should be 1
        normalized = normalize_vector(v)
        assert math.isclose(vector_norm(normalized), 1.0, rel_tol=1e-10)
    
    @given(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=10)
    )
    def test_cosine_similarity_properties(self, v1, v2):
        """Test properties of cosine_similarity function."""
        # Filter out NaN and Inf values
        v1 = [x for x in v1 if not math.isnan(x) and not math.isinf(x)]
        v2 = [x for x in v2 if not math.isnan(x) and not math.isinf(x)]
        if not v1 or not v2:
            return
        
        # Make vectors the same length
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        # Skip zero vectors
        if all(abs(x) < 1e-10 for x in v1) or all(abs(x) < 1e-10 for x in v2):
            return
        
        # The cosine similarity should be between -1 and 1
        similarity = cosine_similarity(v1, v2)
        assert -1.0 <= similarity <= 1.0
        
        # The cosine similarity of a vector with itself should be 1
        assert math.isclose(cosine_similarity(v1, v1), 1.0, rel_tol=1e-10)
        
        # The cosine similarity should be symmetric
        assert math.isclose(cosine_similarity(v1, v2), cosine_similarity(v2, v1), rel_tol=1e-10)


if __name__ == "__main__":
    unittest.main()
