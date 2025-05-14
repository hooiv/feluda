"""
Autonomic Package

This package provides autonomic (self-tuning, self-healing, self-optimizing) capabilities for Feluda.
"""

from feluda.autonomic.ml_tuning import (
    BayesianOptimizer,
    GeneticOptimizer,
    OptimizationAlgorithm,
    OptimizationConfig,
    Optimizer,
    Parameter,
    ParameterType,
    RandomSearchOptimizer,
    create_optimizer,
    optimize_parameters,
)
from feluda.autonomic.self_healing import (
    HealingAction,
    HealingStrategy,
    HealthCheck,
    HealthStatus,
    SelfHealingSystem,
    create_circuit_breaker_health_check,
)

__all__ = [
    # ML-driven tuning
    "OptimizationAlgorithm",
    "ParameterType",
    "Parameter",
    "OptimizationConfig",
    "Optimizer",
    "BayesianOptimizer",
    "GeneticOptimizer",
    "RandomSearchOptimizer",
    "create_optimizer",
    "optimize_parameters",
    
    # Self-healing
    "HealingStrategy",
    "HealingAction",
    "HealthStatus",
    "HealthCheck",
    "SelfHealingSystem",
    "create_circuit_breaker_health_check",
]
