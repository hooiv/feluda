"""
Resilience benchmarks for Feluda.

This module contains benchmarks for the resilience components of Feluda.
Run with: pytest benchmarks/ --benchmark-only
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feluda.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
)


# Define some functions to benchmark
def function_without_circuit_breaker():
    """A function without a circuit breaker."""
    return "success"


@circuit_breaker(
    name="test_circuit_breaker",
    failure_threshold=3,
    recovery_timeout=1.0,
)
def function_with_circuit_breaker():
    """A function with a circuit breaker."""
    return "success"


@circuit_breaker(
    name="failing_circuit_breaker",
    failure_threshold=3,
    recovery_timeout=1.0,
)
def failing_function():
    """A function that fails."""
    raise ValueError("Simulated failure")


# Benchmarks for circuit breaker
def test_function_without_circuit_breaker(benchmark):
    """Benchmark a function without a circuit breaker."""
    benchmark(function_without_circuit_breaker)


def test_function_with_circuit_breaker(benchmark):
    """Benchmark a function with a circuit breaker."""
    # Reset the circuit breaker
    CircuitBreakerRegistry.reset("test_circuit_breaker")
    
    benchmark(function_with_circuit_breaker)


def test_circuit_breaker_creation(benchmark):
    """Benchmark circuit breaker creation."""
    def create_circuit_breaker():
        return CircuitBreaker(
            name="benchmark_circuit_breaker",
            failure_threshold=3,
            recovery_timeout=1.0,
        )
    
    benchmark(create_circuit_breaker)


def test_circuit_breaker_call_success(benchmark):
    """Benchmark circuit breaker call with success."""
    circuit_breaker = CircuitBreaker(
        name="benchmark_circuit_breaker_success",
        failure_threshold=3,
        recovery_timeout=1.0,
    )
    
    def call_with_circuit_breaker():
        return circuit_breaker.call(lambda: "success")
    
    benchmark(call_with_circuit_breaker)


def test_circuit_breaker_call_failure(benchmark):
    """Benchmark circuit breaker call with failure."""
    circuit_breaker = CircuitBreaker(
        name="benchmark_circuit_breaker_failure",
        failure_threshold=3,
        recovery_timeout=1.0,
        expected_exceptions=[ValueError],
    )
    
    def call_with_circuit_breaker():
        try:
            return circuit_breaker.call(lambda: (_ for _ in ()).throw(ValueError("Simulated failure")))
        except ValueError:
            return "caught"
    
    benchmark(call_with_circuit_breaker)


def test_circuit_breaker_state_transition(benchmark):
    """Benchmark circuit breaker state transition."""
    circuit_breaker = CircuitBreaker(
        name="benchmark_circuit_breaker_transition",
        failure_threshold=3,
        recovery_timeout=0.1,
        expected_exceptions=[ValueError],
    )
    
    def transition_state():
        # Force the circuit breaker to open
        with circuit_breaker._state_lock:
            circuit_breaker._state = CircuitState.OPEN
            circuit_breaker._last_failure_time = time.time() - 0.2
        
        # Try to call, which should transition to half-open
        try:
            circuit_breaker.call(lambda: "success")
        except:
            pass
        
        return circuit_breaker.state
    
    benchmark(transition_state)


def test_circuit_breaker_registry_get(benchmark):
    """Benchmark circuit breaker registry get."""
    # Create a circuit breaker
    CircuitBreaker(
        name="benchmark_registry_get",
        failure_threshold=3,
        recovery_timeout=1.0,
    )
    
    def get_from_registry():
        return CircuitBreakerRegistry.get("benchmark_registry_get")
    
    benchmark(get_from_registry)


def test_circuit_breaker_registry_reset(benchmark):
    """Benchmark circuit breaker registry reset."""
    # Create a circuit breaker
    CircuitBreaker(
        name="benchmark_registry_reset",
        failure_threshold=3,
        recovery_timeout=1.0,
    )
    
    def reset_in_registry():
        CircuitBreakerRegistry.reset("benchmark_registry_reset")
    
    benchmark(reset_in_registry)
