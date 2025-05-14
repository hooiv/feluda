"""
Tests for the circuit breaker module.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

from feluda.exceptions import CircuitBreakerError
from feluda.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreaker(unittest.TestCase):
    """Tests for the CircuitBreaker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset all circuit breakers before each test
        CircuitBreakerRegistry.reset_all()
    
    def test_initial_state(self):
        """Test the initial state of a circuit breaker."""
        cb = CircuitBreaker("test")
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.last_failure_time, 0.0)
    
    def test_successful_call(self):
        """Test a successful call."""
        cb = CircuitBreaker("test")
        
        # Define a test function
        def test_func(x, y):
            return x + y
        
        # Call the function through the circuit breaker
        result = cb.call(test_func, 1, 2)
        
        # Check the result and state
        self.assertEqual(result, 3)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
    
    def test_failed_call(self):
        """Test a failed call."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        # Define a test function that raises an exception
        def test_func():
            raise ValueError("Test error")
        
        # Call the function through the circuit breaker
        with self.assertRaises(CircuitBreakerError):
            for _ in range(3):
                cb.call(test_func)
        
        # Check the state
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertEqual(cb.failure_count, 3)
        self.assertGreater(cb.last_failure_time, 0.0)
    
    def test_fallback_function(self):
        """Test the fallback function."""
        # Define a fallback function
        fallback = MagicMock(return_value="fallback")
        
        cb = CircuitBreaker("test", failure_threshold=1, fallback_function=fallback)
        
        # Define a test function that raises an exception
        def test_func():
            raise ValueError("Test error")
        
        # Call the function through the circuit breaker
        result = cb.call(test_func)
        
        # Check that the fallback function was called
        fallback.assert_called_once()
        self.assertEqual(result, "fallback")
    
    def test_recovery(self):
        """Test recovery from an open state."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        
        # Define a test function that raises an exception
        def test_func():
            raise ValueError("Test error")
        
        # Define a successful function
        def success_func():
            return "success"
        
        # Call the function through the circuit breaker to open the circuit
        with self.assertRaises(CircuitBreakerError):
            cb.call(test_func)
        
        # Check the state
        self.assertEqual(cb.state, CircuitState.OPEN)
        
        # Wait for the recovery timeout
        time.sleep(0.2)
        
        # Call a successful function
        result = cb.call(success_func)
        
        # Check that the circuit is now closed
        self.assertEqual(result, "success")
        self.assertEqual(cb.state, CircuitState.CLOSED)
    
    def test_half_open_state(self):
        """Test the half-open state."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        
        # Define a test function that raises an exception
        def test_func():
            raise ValueError("Test error")
        
        # Call the function through the circuit breaker to open the circuit
        with self.assertRaises(CircuitBreakerError):
            cb.call(test_func)
        
        # Check the state
        self.assertEqual(cb.state, CircuitState.OPEN)
        
        # Wait for the recovery timeout
        time.sleep(0.2)
        
        # The next call should transition to HALF_OPEN
        with patch.object(cb, '_state', CircuitState.OPEN):
            with self.assertRaises(CircuitBreakerError):
                cb.call(test_func)
            
            # Check that the circuit is still open
            self.assertEqual(cb.state, CircuitState.OPEN)
    
    def test_expected_exceptions(self):
        """Test that only expected exceptions trigger the circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=1, expected_exceptions=[ValueError])
        
        # Define a test function that raises a ValueError
        def value_error_func():
            raise ValueError("Test error")
        
        # Define a test function that raises a TypeError
        def type_error_func():
            raise TypeError("Test error")
        
        # Call the function that raises a ValueError
        with self.assertRaises(CircuitBreakerError):
            cb.call(value_error_func)
        
        # Check the state
        self.assertEqual(cb.state, CircuitState.OPEN)
        
        # Reset the circuit breaker
        cb.reset()
        
        # Call the function that raises a TypeError
        with self.assertRaises(TypeError):
            cb.call(type_error_func)
        
        # Check that the circuit is still closed
        self.assertEqual(cb.state, CircuitState.CLOSED)
    
    def test_decorator(self):
        """Test the circuit_breaker decorator."""
        # Define a test function with the decorator
        @circuit_breaker("test_decorator", failure_threshold=1)
        def test_func():
            raise ValueError("Test error")
        
        # Call the function
        with self.assertRaises(CircuitBreakerError):
            test_func()
        
        # Check that the circuit breaker was registered
        cb = CircuitBreakerRegistry.get("test_decorator")
        self.assertIsNotNone(cb)
        self.assertEqual(cb.state, CircuitState.OPEN)


class TestCircuitBreakerRegistry(unittest.TestCase):
    """Tests for the CircuitBreakerRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset all circuit breakers before each test
        CircuitBreakerRegistry.reset_all()
    
    def test_register(self):
        """Test registering a circuit breaker."""
        cb = CircuitBreaker("test_register")
        
        # Check that the circuit breaker was registered
        registered_cb = CircuitBreakerRegistry.get("test_register")
        self.assertIs(registered_cb, cb)
    
    def test_register_duplicate(self):
        """Test registering a circuit breaker with a duplicate name."""
        CircuitBreaker("test_duplicate")
        
        # Try to register another circuit breaker with the same name
        with self.assertRaises(ValueError):
            CircuitBreaker("test_duplicate")
    
    def test_get_all(self):
        """Test getting all registered circuit breakers."""
        cb1 = CircuitBreaker("test1")
        cb2 = CircuitBreaker("test2")
        
        # Get all circuit breakers
        all_cbs = CircuitBreakerRegistry.get_all()
        
        # Check that both circuit breakers are in the registry
        self.assertIn("test1", all_cbs)
        self.assertIn("test2", all_cbs)
        self.assertIs(all_cbs["test1"], cb1)
        self.assertIs(all_cbs["test2"], cb2)
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        # Create two circuit breakers
        cb1 = CircuitBreaker("test1", failure_threshold=1)
        cb2 = CircuitBreaker("test2", failure_threshold=1)
        
        # Define a test function that raises an exception
        def test_func():
            raise ValueError("Test error")
        
        # Open both circuit breakers
        with self.assertRaises(CircuitBreakerError):
            cb1.call(test_func)
        
        with self.assertRaises(CircuitBreakerError):
            cb2.call(test_func)
        
        # Check that both circuit breakers are open
        self.assertEqual(cb1.state, CircuitState.OPEN)
        self.assertEqual(cb2.state, CircuitState.OPEN)
        
        # Reset all circuit breakers
        CircuitBreakerRegistry.reset_all()
        
        # Check that both circuit breakers are closed
        self.assertEqual(cb1.state, CircuitState.CLOSED)
        self.assertEqual(cb2.state, CircuitState.CLOSED)


if __name__ == "__main__":
    unittest.main()
