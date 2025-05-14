"""
Autonomic benchmarks for Feluda.

This module contains benchmarks for the autonomic components of Feluda.
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

from feluda.autonomic.ml_tuning import (
    OptimizationAlgorithm,
    OptimizationConfig,
    Parameter,
    ParameterType,
    RandomSearchOptimizer,
    create_optimizer,
)
from feluda.autonomic.self_healing import (
    HealingAction,
    HealingStrategy,
    HealthCheck,
    HealthStatus,
    SelfHealingSystem,
)


# Define some fixtures
@pytest.fixture
def optimization_config():
    """Create an optimization configuration."""
    params = [
        Parameter(
            name="x",
            parameter_type=ParameterType.CONTINUOUS,
            min_value=-5.0,
            max_value=5.0,
        ),
        Parameter(
            name="y",
            parameter_type=ParameterType.CONTINUOUS,
            min_value=-5.0,
            max_value=5.0,
        ),
    ]
    
    return OptimizationConfig(
        algorithm=OptimizationAlgorithm.RANDOM_SEARCH,
        parameters=params,
        max_iterations=10,
        random_seed=42,
    )


@pytest.fixture
def objective_function():
    """Create an objective function."""
    def func(params):
        return -((params["x"] ** 2) + (params["y"] ** 2))
    
    return func


@pytest.fixture
def self_healing_system():
    """Create a self-healing system."""
    system = SelfHealingSystem()
    
    # Create a health check
    def check_function():
        return HealthStatus.HEALTHY
    
    healing_actions = {
        HealthStatus.DEGRADED: [
            HealingAction(
                strategy=HealingStrategy.RETRY,
                params={"max_retries": 3, "retry_delay": 0.1},
            ),
        ],
        HealthStatus.UNHEALTHY: [
            HealingAction(
                strategy=HealingStrategy.RESTART,
                params={"restart_function": lambda: None},
            ),
        ],
    }
    
    health_check = HealthCheck(
        name="test",
        check_function=check_function,
        healing_actions=healing_actions,
        check_interval=0.1,
    )
    
    system.add_health_check(health_check)
    
    return system


# Benchmarks for ML tuning
def test_parameter_sample(benchmark):
    """Benchmark parameter sampling."""
    param = Parameter(
        name="x",
        parameter_type=ParameterType.CONTINUOUS,
        min_value=-5.0,
        max_value=5.0,
    )
    
    benchmark(param.sample)


def test_create_optimizer(benchmark, optimization_config):
    """Benchmark optimizer creation."""
    benchmark(create_optimizer, optimization_config)


def test_random_search_optimizer_optimize(benchmark, optimization_config, objective_function):
    """Benchmark random search optimization."""
    optimizer = RandomSearchOptimizer(optimization_config)
    
    benchmark(optimizer.optimize, objective_function)


# Benchmarks for self-healing
def test_health_check_should_check(benchmark):
    """Benchmark health check should_check."""
    def check_function():
        return HealthStatus.HEALTHY
    
    health_check = HealthCheck(
        name="test",
        check_function=check_function,
        healing_actions={},
        check_interval=0.1,
    )
    
    benchmark(health_check.should_check)


def test_health_check_check(benchmark):
    """Benchmark health check check."""
    def check_function():
        return HealthStatus.HEALTHY
    
    health_check = HealthCheck(
        name="test",
        check_function=check_function,
        healing_actions={},
        check_interval=0.1,
    )
    
    benchmark(health_check.check)


def test_health_check_get_healing_actions(benchmark):
    """Benchmark health check get_healing_actions."""
    def check_function():
        return HealthStatus.HEALTHY
    
    healing_actions = {
        HealthStatus.DEGRADED: [
            HealingAction(
                strategy=HealingStrategy.RETRY,
                params={"max_retries": 3, "retry_delay": 0.1},
            ),
        ],
        HealthStatus.UNHEALTHY: [
            HealingAction(
                strategy=HealingStrategy.RESTART,
                params={"restart_function": lambda: None},
            ),
        ],
    }
    
    health_check = HealthCheck(
        name="test",
        check_function=check_function,
        healing_actions=healing_actions,
        check_interval=0.1,
    )
    
    benchmark(health_check.get_healing_actions)


def test_self_healing_system_check_health(benchmark, self_healing_system):
    """Benchmark self-healing system check_health."""
    benchmark(self_healing_system.check_health)


def test_self_healing_system_heal(benchmark, self_healing_system):
    """Benchmark self-healing system heal."""
    benchmark(self_healing_system.heal)
