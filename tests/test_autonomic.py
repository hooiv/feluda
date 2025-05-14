"""
Unit tests for the autonomic module.
"""

import unittest
from unittest import mock
import time

import numpy as np

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
from feluda.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState


class TestMLTuningModule(unittest.TestCase):
    """Test cases for the ML tuning module."""
    
    def test_parameter(self):
        """Test the Parameter class."""
        # Test continuous parameter
        param = Parameter(
            name="learning_rate",
            parameter_type=ParameterType.CONTINUOUS,
            min_value=0.001,
            max_value=0.1,
            default=0.01,
        )
        
        self.assertEqual(param.name, "learning_rate")
        self.assertEqual(param.parameter_type, ParameterType.CONTINUOUS)
        self.assertEqual(param.min_value, 0.001)
        self.assertEqual(param.max_value, 0.1)
        self.assertEqual(param.default, 0.01)
        
        # Test sampling
        value = param.sample()
        self.assertGreaterEqual(value, param.min_value)
        self.assertLessEqual(value, param.max_value)
        
        # Test to_dict and from_dict
        param_dict = param.to_dict()
        param2 = Parameter.from_dict(param_dict)
        
        self.assertEqual(param.name, param2.name)
        self.assertEqual(param.parameter_type, param2.parameter_type)
        self.assertEqual(param.min_value, param2.min_value)
        self.assertEqual(param.max_value, param2.max_value)
        self.assertEqual(param.default, param2.default)
        
        # Test discrete parameter
        param = Parameter(
            name="batch_size",
            parameter_type=ParameterType.DISCRETE,
            min_value=16,
            max_value=128,
            default=32,
        )
        
        self.assertEqual(param.name, "batch_size")
        self.assertEqual(param.parameter_type, ParameterType.DISCRETE)
        
        # Test sampling
        value = param.sample()
        self.assertGreaterEqual(value, param.min_value)
        self.assertLessEqual(value, param.max_value)
        self.assertEqual(int(value), value)  # Should be an integer
        
        # Test categorical parameter
        param = Parameter(
            name="activation",
            parameter_type=ParameterType.CATEGORICAL,
            choices=["relu", "sigmoid", "tanh"],
            default="relu",
        )
        
        self.assertEqual(param.name, "activation")
        self.assertEqual(param.parameter_type, ParameterType.CATEGORICAL)
        self.assertEqual(param.choices, ["relu", "sigmoid", "tanh"])
        self.assertEqual(param.default, "relu")
        
        # Test sampling
        value = param.sample()
        self.assertIn(value, param.choices)
        
        # Test boolean parameter
        param = Parameter(
            name="use_bias",
            parameter_type=ParameterType.BOOLEAN,
            default=True,
        )
        
        self.assertEqual(param.name, "use_bias")
        self.assertEqual(param.parameter_type, ParameterType.BOOLEAN)
        self.assertEqual(param.default, True)
        
        # Test sampling
        value = param.sample()
        self.assertIn(value, [True, False])
        
        # Test validation
        with self.assertRaises(ValueError):
            # Min value >= max value
            Parameter(
                name="invalid",
                parameter_type=ParameterType.CONTINUOUS,
                min_value=0.1,
                max_value=0.1,
            )
        
        with self.assertRaises(ValueError):
            # Categorical without choices
            Parameter(
                name="invalid",
                parameter_type=ParameterType.CATEGORICAL,
            )
        
        with self.assertRaises(ValueError):
            # Default outside range
            Parameter(
                name="invalid",
                parameter_type=ParameterType.CONTINUOUS,
                min_value=0.001,
                max_value=0.1,
                default=0.2,
            )
    
    def test_optimization_config(self):
        """Test the OptimizationConfig class."""
        # Create parameters
        params = [
            Parameter(
                name="learning_rate",
                parameter_type=ParameterType.CONTINUOUS,
                min_value=0.001,
                max_value=0.1,
                default=0.01,
            ),
            Parameter(
                name="batch_size",
                parameter_type=ParameterType.DISCRETE,
                min_value=16,
                max_value=128,
                default=32,
            ),
        ]
        
        # Create a config
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.BAYESIAN,
            parameters=params,
            max_iterations=100,
            random_seed=42,
            algorithm_params={"n_initial_points": 10},
        )
        
        # Check the attributes
        self.assertEqual(config.algorithm, OptimizationAlgorithm.BAYESIAN)
        self.assertEqual(config.parameters, params)
        self.assertEqual(config.max_iterations, 100)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.algorithm_params, {"n_initial_points": 10})
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        config2 = OptimizationConfig.from_dict(config_dict)
        
        self.assertEqual(config.algorithm, config2.algorithm)
        self.assertEqual(len(config.parameters), len(config2.parameters))
        self.assertEqual(config.max_iterations, config2.max_iterations)
        self.assertEqual(config.random_seed, config2.random_seed)
        self.assertEqual(config.algorithm_params, config2.algorithm_params)
    
    def test_random_search_optimizer(self):
        """Test the RandomSearchOptimizer class."""
        # Create parameters
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
        
        # Create a config
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.RANDOM_SEARCH,
            parameters=params,
            max_iterations=10,
            random_seed=42,
        )
        
        # Create an optimizer
        optimizer = RandomSearchOptimizer(config)
        
        # Define an objective function (minimize x^2 + y^2)
        def objective_function(params):
            return -((params["x"] ** 2) + (params["y"] ** 2))
        
        # Optimize
        best_params = optimizer.optimize(objective_function)
        
        # Check the result
        self.assertIn("x", best_params)
        self.assertIn("y", best_params)
        self.assertIsNotNone(optimizer.best_score)
        self.assertEqual(len(optimizer.history), 10)
    
    def test_create_optimizer(self):
        """Test the create_optimizer function."""
        # Create parameters
        params = [
            Parameter(
                name="x",
                parameter_type=ParameterType.CONTINUOUS,
                min_value=-5.0,
                max_value=5.0,
            ),
        ]
        
        # Test creating a random search optimizer
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.RANDOM_SEARCH,
            parameters=params,
        )
        
        optimizer = create_optimizer(config)
        self.assertIsInstance(optimizer, RandomSearchOptimizer)
        
        # Test creating a genetic optimizer
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC,
            parameters=params,
        )
        
        optimizer = create_optimizer(config)
        self.assertIsInstance(optimizer, GeneticOptimizer)
        
        # Test creating a Bayesian optimizer
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.BAYESIAN,
            parameters=params,
        )
        
        # This will fail if scikit-optimize is not installed
        try:
            optimizer = create_optimizer(config)
            self.assertIsInstance(optimizer, BayesianOptimizer)
        except ImportError:
            pass
        
        # Test unsupported algorithm
        config = OptimizationConfig(
            algorithm="unsupported",
            parameters=params,
        )
        
        with self.assertRaises(ValueError):
            create_optimizer(config)
    
    def test_optimize_parameters(self):
        """Test the optimize_parameters function."""
        # Create parameters
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
        
        # Create a config
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.RANDOM_SEARCH,
            parameters=params,
            max_iterations=10,
            random_seed=42,
        )
        
        # Define an objective function (minimize x^2 + y^2)
        def objective_function(params):
            return -((params["x"] ** 2) + (params["y"] ** 2))
        
        # Optimize
        with mock.patch("feluda.autonomic.ml_tuning.create_optimizer") as mock_create:
            mock_optimizer = mock.MagicMock()
            mock_optimizer.optimize.return_value = {"x": 0.0, "y": 0.0}
            mock_create.return_value = mock_optimizer
            
            best_params = optimize_parameters(objective_function, config)
            
            mock_create.assert_called_once_with(config)
            mock_optimizer.optimize.assert_called_once_with(objective_function)
            
            self.assertEqual(best_params, {"x": 0.0, "y": 0.0})


class TestSelfHealingModule(unittest.TestCase):
    """Test cases for the self-healing module."""
    
    def test_healing_action(self):
        """Test the HealingAction class."""
        # Create an action
        action = HealingAction(
            strategy=HealingStrategy.RETRY,
            params={"max_retries": 3, "retry_delay": 1.0},
        )
        
        # Check the attributes
        self.assertEqual(action.strategy, HealingStrategy.RETRY)
        self.assertEqual(action.params, {"max_retries": 3, "retry_delay": 1.0})
        
        # Test to_dict and from_dict
        action_dict = action.to_dict()
        action2 = HealingAction.from_dict(action_dict)
        
        self.assertEqual(action.strategy, action2.strategy)
        self.assertEqual(action.params, action2.params)
    
    def test_health_check(self):
        """Test the HealthCheck class."""
        # Create a check function
        def check_function():
            return HealthStatus.HEALTHY
        
        # Create healing actions
        healing_actions = {
            HealthStatus.DEGRADED: [
                HealingAction(
                    strategy=HealingStrategy.RETRY,
                    params={"max_retries": 3, "retry_delay": 1.0},
                ),
            ],
            HealthStatus.UNHEALTHY: [
                HealingAction(
                    strategy=HealingStrategy.RESTART,
                    params={"restart_function": lambda: None},
                ),
            ],
        }
        
        # Create a health check
        health_check = HealthCheck(
            name="test",
            check_function=check_function,
            healing_actions=healing_actions,
            check_interval=60.0,
        )
        
        # Check the attributes
        self.assertEqual(health_check.name, "test")
        self.assertEqual(health_check.check_function, check_function)
        self.assertEqual(health_check.healing_actions, healing_actions)
        self.assertEqual(health_check.check_interval, 60.0)
        self.assertEqual(health_check.last_status, HealthStatus.UNKNOWN)
        
        # Test should_check
        self.assertTrue(health_check.should_check())
        
        # Test check
        status = health_check.check()
        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertEqual(health_check.last_status, HealthStatus.HEALTHY)
        
        # Test get_healing_actions
        actions = health_check.get_healing_actions()
        self.assertEqual(actions, [])  # No actions for HEALTHY status
        
        # Test to_dict
        check_dict = health_check.to_dict()
        self.assertEqual(check_dict["name"], "test")
        self.assertEqual(check_dict["check_interval"], 60.0)
        self.assertEqual(check_dict["last_status"], HealthStatus.HEALTHY)
    
    def test_self_healing_system(self):
        """Test the SelfHealingSystem class."""
        # Create a system
        system = SelfHealingSystem()
        
        # Create a health check
        def check_function():
            return HealthStatus.HEALTHY
        
        healing_actions = {
            HealthStatus.DEGRADED: [
                HealingAction(
                    strategy=HealingStrategy.RETRY,
                    params={"max_retries": 3, "retry_delay": 1.0},
                ),
            ],
        }
        
        health_check = HealthCheck(
            name="test",
            check_function=check_function,
            healing_actions=healing_actions,
            check_interval=60.0,
        )
        
        # Add the health check
        system.add_health_check(health_check)
        
        # Check that it was added
        self.assertIn("test", system.health_checks)
        
        # Test check_health
        health = system.check_health()
        self.assertIn("test", health)
        self.assertEqual(health["test"], HealthStatus.HEALTHY)
        
        # Test heal
        healing_results = system.heal()
        self.assertIn("test", healing_results)
        self.assertEqual(healing_results["test"], [])  # No actions for HEALTHY status
        
        # Test remove_health_check
        system.remove_health_check("test")
        self.assertNotIn("test", system.health_checks)
    
    @mock.patch("feluda.resilience.circuit_breaker.CircuitBreakerRegistry")
    def test_create_circuit_breaker_health_check(self, mock_registry):
        """Test the create_circuit_breaker_health_check function."""
        # Mock the registry
        mock_cb = mock.MagicMock()
        mock_cb.state = CircuitState.CLOSED
        mock_registry.get.return_value = mock_cb
        
        # Create a health check
        health_check = create_circuit_breaker_health_check(
            name="test",
            circuit_breaker_name="test_cb",
            check_interval=60.0,
        )
        
        # Check the attributes
        self.assertEqual(health_check.name, "test")
        self.assertEqual(health_check.check_interval, 60.0)
        
        # Test the check function
        status = health_check.check_function()
        self.assertEqual(status, HealthStatus.HEALTHY)
        
        # Test with different circuit breaker states
        mock_cb.state = CircuitState.HALF_OPEN
        status = health_check.check_function()
        self.assertEqual(status, HealthStatus.DEGRADED)
        
        mock_cb.state = CircuitState.OPEN
        status = health_check.check_function()
        self.assertEqual(status, HealthStatus.UNHEALTHY)
        
        # Test healing actions
        self.assertIn(HealthStatus.DEGRADED, health_check.healing_actions)
        self.assertIn(HealthStatus.UNHEALTHY, health_check.healing_actions)
        
        degraded_actions = health_check.healing_actions[HealthStatus.DEGRADED]
        self.assertEqual(len(degraded_actions), 1)
        self.assertEqual(degraded_actions[0].strategy, HealingStrategy.RETRY)
        
        unhealthy_actions = health_check.healing_actions[HealthStatus.UNHEALTHY]
        self.assertEqual(len(unhealthy_actions), 2)
        self.assertEqual(unhealthy_actions[0].strategy, HealingStrategy.FALLBACK)
        self.assertEqual(unhealthy_actions[1].strategy, HealingStrategy.RECONFIGURE)


if __name__ == "__main__":
    unittest.main()
