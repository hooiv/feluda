"""
ML-Driven Tuning Module

This module provides ML-driven tuning capabilities for Feluda.
"""

import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class OptimizationAlgorithm(str, Enum):
    """Enum for optimization algorithms."""
    
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    GRADIENT_DESCENT = "gradient_descent"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ParameterType(str, Enum):
    """Enum for parameter types."""
    
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Parameter:
    """
    Parameter for optimization.
    
    This class represents a parameter that can be optimized.
    """
    
    def __init__(
        self,
        name: str,
        parameter_type: ParameterType,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        choices: Optional[List[Any]] = None,
        default: Optional[Any] = None,
    ):
        """
        Initialize a Parameter.
        
        Args:
            name: The name of the parameter.
            parameter_type: The type of the parameter.
            min_value: The minimum value for continuous and discrete parameters.
            max_value: The maximum value for continuous and discrete parameters.
            choices: The choices for categorical parameters.
            default: The default value for the parameter.
        """
        self.name = name
        self.parameter_type = parameter_type
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices
        self.default = default
        
        # Validate the parameter
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate the parameter.
        
        Raises:
            ValueError: If the parameter is invalid.
        """
        if self.parameter_type in [ParameterType.CONTINUOUS, ParameterType.DISCRETE]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Continuous and discrete parameters must have min_value and max_value: {self.name}")
            if self.min_value >= self.max_value:
                raise ValueError(f"min_value must be less than max_value: {self.name}")
        
        if self.parameter_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Categorical parameters must have choices: {self.name}")
        
        if self.default is not None:
            if self.parameter_type in [ParameterType.CONTINUOUS, ParameterType.DISCRETE]:
                if self.default < self.min_value or self.default > self.max_value:
                    raise ValueError(f"Default value must be within min_value and max_value: {self.name}")
            elif self.parameter_type == ParameterType.CATEGORICAL:
                if self.default not in self.choices:
                    raise ValueError(f"Default value must be one of the choices: {self.name}")
            elif self.parameter_type == ParameterType.BOOLEAN:
                if not isinstance(self.default, bool):
                    raise ValueError(f"Default value must be a boolean: {self.name}")
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Any:
        """
        Sample a value from the parameter.
        
        Args:
            random_state: The random state to use for sampling.
            
        Returns:
            A sampled value.
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        if self.parameter_type == ParameterType.CONTINUOUS:
            return random_state.uniform(self.min_value, self.max_value)
        elif self.parameter_type == ParameterType.DISCRETE:
            return random_state.randint(self.min_value, self.max_value + 1)
        elif self.parameter_type == ParameterType.CATEGORICAL:
            return random_state.choice(self.choices)
        elif self.parameter_type == ParameterType.BOOLEAN:
            return random_state.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {self.parameter_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the parameter to a dictionary.
        
        Returns:
            A dictionary representation of the parameter.
        """
        return {
            "name": self.name,
            "parameter_type": self.parameter_type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "choices": self.choices,
            "default": self.default,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Parameter":
        """
        Create a parameter from a dictionary.
        
        Args:
            data: The dictionary representation of the parameter.
            
        Returns:
            The created parameter.
        """
        return cls(
            name=data["name"],
            parameter_type=data["parameter_type"],
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            choices=data.get("choices"),
            default=data.get("default"),
        )


class OptimizationConfig:
    """
    Configuration for optimization.
    
    This class holds the configuration for optimization, including the algorithm,
    parameters, and objective function.
    """
    
    def __init__(
        self,
        algorithm: OptimizationAlgorithm,
        parameters: List[Parameter],
        max_iterations: int = 100,
        random_seed: Optional[int] = None,
        algorithm_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an OptimizationConfig.
        
        Args:
            algorithm: The optimization algorithm to use.
            parameters: The parameters to optimize.
            max_iterations: The maximum number of iterations.
            random_seed: The random seed to use.
            algorithm_params: Additional parameters for the algorithm.
        """
        self.algorithm = algorithm
        self.parameters = parameters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.algorithm_params = algorithm_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            A dictionary representation of the configuration.
        """
        return {
            "algorithm": self.algorithm,
            "parameters": [param.to_dict() for param in self.parameters],
            "max_iterations": self.max_iterations,
            "random_seed": self.random_seed,
            "algorithm_params": self.algorithm_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            data: The dictionary representation of the configuration.
            
        Returns:
            The created configuration.
        """
        return cls(
            algorithm=data["algorithm"],
            parameters=[Parameter.from_dict(param) for param in data["parameters"]],
            max_iterations=data.get("max_iterations", 100),
            random_seed=data.get("random_seed"),
            algorithm_params=data.get("algorithm_params", {}),
        )


class Optimizer:
    """
    Base class for optimizers.
    
    This class defines the interface for optimizers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize an Optimizer.
        
        Args:
            config: The optimization configuration.
        """
        self.config = config
        self.random_state = np.random.RandomState(config.random_seed)
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
    
    def optimize(self, objective_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Optimize the objective function.
        
        Args:
            objective_function: The objective function to optimize.
            
        Returns:
            The best parameters found.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement optimize")
    
    def _update_best(self, params: Dict[str, Any], score: float) -> None:
        """
        Update the best parameters and score.
        
        Args:
            params: The parameters to evaluate.
            score: The score of the parameters.
        """
        if self.best_score is None or score > self.best_score:
            self.best_params = params.copy()
            self.best_score = score
        
        self.history.append({
            "params": params.copy(),
            "score": score,
        })


class BayesianOptimizer(Optimizer):
    """
    Bayesian optimization.
    
    This class implements Bayesian optimization using Gaussian processes.
    """
    
    def optimize(self, objective_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Optimize the objective function using Bayesian optimization.
        
        Args:
            objective_function: The objective function to optimize.
            
        Returns:
            The best parameters found.
        """
        try:
            from skopt import Optimizer as SkOptimizer
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            log.error("scikit-optimize is not installed")
            raise ImportError("scikit-optimize is not installed. Install it with 'pip install scikit-optimize'.")
        
        # Create the search space
        space = []
        param_names = []
        
        for param in self.config.parameters:
            param_names.append(param.name)
            
            if param.parameter_type == ParameterType.CONTINUOUS:
                space.append(Real(param.min_value, param.max_value))
            elif param.parameter_type == ParameterType.DISCRETE:
                space.append(Integer(param.min_value, param.max_value))
            elif param.parameter_type == ParameterType.CATEGORICAL:
                space.append(Categorical(param.choices))
            elif param.parameter_type == ParameterType.BOOLEAN:
                space.append(Categorical([True, False]))
        
        # Create the optimizer
        n_initial_points = self.config.algorithm_params.get("n_initial_points", 10)
        skopt = SkOptimizer(space, random_state=self.random_state, n_initial_points=n_initial_points)
        
        # Run the optimization
        for i in range(self.config.max_iterations):
            # Get the next point to evaluate
            x = skopt.ask()
            
            # Convert to a dictionary
            params = {name: value for name, value in zip(param_names, x)}
            
            # Evaluate the objective function
            score = objective_function(params)
            
            # Update the optimizer
            skopt.tell(x, -score)  # Minimize the negative score
            
            # Update the best parameters
            self._update_best(params, score)
            
            log.info(f"Iteration {i+1}/{self.config.max_iterations}: score={score}, best_score={self.best_score}")
        
        return self.best_params


class GeneticOptimizer(Optimizer):
    """
    Genetic optimization.
    
    This class implements genetic optimization using a genetic algorithm.
    """
    
    def optimize(self, objective_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Optimize the objective function using a genetic algorithm.
        
        Args:
            objective_function: The objective function to optimize.
            
        Returns:
            The best parameters found.
        """
        # Get algorithm parameters
        population_size = self.config.algorithm_params.get("population_size", 50)
        mutation_rate = self.config.algorithm_params.get("mutation_rate", 0.1)
        crossover_rate = self.config.algorithm_params.get("crossover_rate", 0.8)
        
        # Initialize the population
        population = []
        for _ in range(population_size):
            individual = {}
            for param in self.config.parameters:
                individual[param.name] = param.sample(self.random_state)
            population.append(individual)
        
        # Evaluate the initial population
        scores = []
        for individual in population:
            score = objective_function(individual)
            scores.append(score)
            self._update_best(individual, score)
        
        # Run the optimization
        for i in range(self.config.max_iterations):
            # Select parents
            parents = self._select_parents(population, scores)
            
            # Create offspring
            offspring = []
            for j in range(0, len(parents), 2):
                if j + 1 < len(parents):
                    parent1 = parents[j]
                    parent2 = parents[j + 1]
                    
                    if self.random_state.random() < crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    child1 = self._mutate(child1, mutation_rate)
                    child2 = self._mutate(child2, mutation_rate)
                    
                    offspring.append(child1)
                    offspring.append(child2)
            
            # Evaluate the offspring
            offspring_scores = []
            for individual in offspring:
                score = objective_function(individual)
                offspring_scores.append(score)
                self._update_best(individual, score)
            
            # Replace the population
            population = offspring
            scores = offspring_scores
            
            log.info(f"Iteration {i+1}/{self.config.max_iterations}: best_score={self.best_score}")
        
        return self.best_params
    
    def _select_parents(self, population: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction.
        
        Args:
            population: The population to select from.
            scores: The scores of the individuals.
            
        Returns:
            The selected parents.
        """
        # Tournament selection
        tournament_size = self.config.algorithm_params.get("tournament_size", 3)
        parents = []
        
        for _ in range(len(population)):
            # Select tournament_size individuals randomly
            tournament_indices = self.random_state.choice(len(population), tournament_size, replace=False)
            tournament_scores = [scores[i] for i in tournament_indices]
            
            # Select the best individual from the tournament
            winner_index = tournament_indices[np.argmax(tournament_scores)]
            parents.append(population[winner_index].copy())
        
        return parents
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: The first parent.
            parent2: The second parent.
            
        Returns:
            The two children.
        """
        child1 = {}
        child2 = {}
        
        for param in self.config.parameters:
            # Randomly choose which parent to inherit from
            if self.random_state.random() < 0.5:
                child1[param.name] = parent1[param.name]
                child2[param.name] = parent2[param.name]
            else:
                child1[param.name] = parent2[param.name]
                child2[param.name] = parent1[param.name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """
        Mutate an individual.
        
        Args:
            individual: The individual to mutate.
            mutation_rate: The mutation rate.
            
        Returns:
            The mutated individual.
        """
        mutated = individual.copy()
        
        for param in self.config.parameters:
            if self.random_state.random() < mutation_rate:
                mutated[param.name] = param.sample(self.random_state)
        
        return mutated


class RandomSearchOptimizer(Optimizer):
    """
    Random search optimization.
    
    This class implements random search optimization.
    """
    
    def optimize(self, objective_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Optimize the objective function using random search.
        
        Args:
            objective_function: The objective function to optimize.
            
        Returns:
            The best parameters found.
        """
        for i in range(self.config.max_iterations):
            # Sample random parameters
            params = {}
            for param in self.config.parameters:
                params[param.name] = param.sample(self.random_state)
            
            # Evaluate the objective function
            score = objective_function(params)
            
            # Update the best parameters
            self._update_best(params, score)
            
            log.info(f"Iteration {i+1}/{self.config.max_iterations}: score={score}, best_score={self.best_score}")
        
        return self.best_params


def create_optimizer(config: OptimizationConfig) -> Optimizer:
    """
    Create an optimizer based on the configuration.
    
    Args:
        config: The optimization configuration.
        
    Returns:
        An optimizer instance.
        
    Raises:
        ValueError: If the algorithm is not supported.
    """
    if config.algorithm == OptimizationAlgorithm.BAYESIAN:
        return BayesianOptimizer(config)
    elif config.algorithm == OptimizationAlgorithm.GENETIC:
        return GeneticOptimizer(config)
    elif config.algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
        return RandomSearchOptimizer(config)
    else:
        raise ValueError(f"Unsupported optimization algorithm: {config.algorithm}")


def optimize_parameters(
    objective_function: Callable[[Dict[str, Any]], float],
    config: OptimizationConfig,
) -> Dict[str, Any]:
    """
    Optimize parameters using the specified algorithm.
    
    Args:
        objective_function: The objective function to optimize.
        config: The optimization configuration.
        
    Returns:
        The best parameters found.
    """
    optimizer = create_optimizer(config)
    return optimizer.optimize(objective_function)
