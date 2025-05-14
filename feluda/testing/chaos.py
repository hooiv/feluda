"""
Chaos Testing Module

This module provides tools for chaos testing in Feluda.
Chaos testing involves deliberately introducing failures to test system resilience.
"""

import logging
import random
import time
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class FailureMode(str, Enum):
    """Enum for failure modes."""
    
    EXCEPTION = "exception"
    DELAY = "delay"
    CORRUPT_DATA = "corrupt_data"
    MEMORY_LEAK = "memory_leak"
    CPU_LOAD = "cpu_load"
    NETWORK_PARTITION = "network_partition"


class ChaosConfig:
    """
    Configuration for chaos testing.
    
    This class holds the configuration for chaos testing, including the probability
    of failures and the types of failures to introduce.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        failure_probability: float = 0.05,
        enabled_failure_modes: Optional[List[FailureMode]] = None,
        exception_types: Optional[List[Type[Exception]]] = None,
        max_delay_ms: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize a ChaosConfig.
        
        Args:
            enabled: Whether chaos testing is enabled.
            failure_probability: The probability of introducing a failure (0.0 to 1.0).
            enabled_failure_modes: The types of failures to introduce. If None, all types are enabled.
            exception_types: The types of exceptions to raise. If None, ValueError is used.
            max_delay_ms: The maximum delay to introduce in milliseconds.
            seed: The random seed to use. If None, a random seed is used.
        """
        self.enabled = enabled
        self.failure_probability = failure_probability
        self.enabled_failure_modes = enabled_failure_modes or list(FailureMode)
        self.exception_types = exception_types or [ValueError]
        self.max_delay_ms = max_delay_ms
        self.random = random.Random(seed)
    
    def should_fail(self) -> bool:
        """
        Determine whether to introduce a failure.
        
        Returns:
            True if a failure should be introduced, False otherwise.
        """
        return self.enabled and self.random.random() < self.failure_probability
    
    def get_failure_mode(self) -> FailureMode:
        """
        Get a random failure mode.
        
        Returns:
            A random failure mode from the enabled failure modes.
        """
        return self.random.choice(self.enabled_failure_modes)
    
    def get_exception_type(self) -> Type[Exception]:
        """
        Get a random exception type.
        
        Returns:
            A random exception type from the configured exception types.
        """
        return self.random.choice(self.exception_types)
    
    def get_delay_ms(self) -> int:
        """
        Get a random delay in milliseconds.
        
        Returns:
            A random delay between 0 and max_delay_ms.
        """
        return self.random.randint(0, self.max_delay_ms)


# Global chaos configuration
_chaos_config = ChaosConfig()


def get_chaos_config() -> ChaosConfig:
    """
    Get the global chaos configuration.
    
    Returns:
        The global chaos configuration.
    """
    return _chaos_config


def set_chaos_config(config: ChaosConfig) -> None:
    """
    Set the global chaos configuration.
    
    Args:
        config: The chaos configuration to set.
    """
    global _chaos_config
    _chaos_config = config


def chaos_monkey(
    failure_probability: Optional[float] = None,
    enabled_failure_modes: Optional[List[FailureMode]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to introduce chaos into a function.
    
    Args:
        failure_probability: The probability of introducing a failure (0.0 to 1.0).
                            If None, the global configuration is used.
        enabled_failure_modes: The types of failures to introduce.
                              If None, the global configuration is used.
        
    Returns:
        The decorated function.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            config = get_chaos_config()
            
            # Override configuration if specified
            local_failure_probability = failure_probability if failure_probability is not None else config.failure_probability
            local_enabled_failure_modes = enabled_failure_modes if enabled_failure_modes is not None else config.enabled_failure_modes
            
            # Check if we should introduce a failure
            if config.enabled and random.random() < local_failure_probability:
                # Choose a failure mode
                failure_mode = random.choice(local_enabled_failure_modes)
                
                # Introduce the failure
                if failure_mode == FailureMode.EXCEPTION:
                    exception_type = config.get_exception_type()
                    log.warning(f"Chaos monkey introducing exception: {exception_type.__name__}")
                    raise exception_type(f"Chaos monkey exception in {func.__name__}")
                
                elif failure_mode == FailureMode.DELAY:
                    delay_ms = config.get_delay_ms()
                    log.warning(f"Chaos monkey introducing delay: {delay_ms}ms")
                    time.sleep(delay_ms / 1000.0)
                
                elif failure_mode == FailureMode.CORRUPT_DATA:
                    log.warning("Chaos monkey corrupting data")
                    # This is a placeholder for data corruption
                    # In a real implementation, this would modify the function's arguments
                    pass
                
                elif failure_mode == FailureMode.MEMORY_LEAK:
                    log.warning("Chaos monkey introducing memory leak")
                    # This is a placeholder for memory leaks
                    # In a real implementation, this would allocate memory that is not freed
                    pass
                
                elif failure_mode == FailureMode.CPU_LOAD:
                    log.warning("Chaos monkey introducing CPU load")
                    # This is a placeholder for CPU load
                    # In a real implementation, this would perform CPU-intensive operations
                    pass
                
                elif failure_mode == FailureMode.NETWORK_PARTITION:
                    log.warning("Chaos monkey introducing network partition")
                    # This is a placeholder for network partitions
                    # In a real implementation, this would block network connections
                    pass
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


@contextmanager
def chaos_context(
    enabled: bool = True,
    failure_probability: float = 0.05,
    enabled_failure_modes: Optional[List[FailureMode]] = None,
    exception_types: Optional[List[Type[Exception]]] = None,
    max_delay_ms: int = 1000,
    seed: Optional[int] = None,
):
    """
    Context manager for chaos testing.
    
    This context manager enables chaos testing within its scope and restores
    the original configuration when exiting the scope.
    
    Args:
        enabled: Whether chaos testing is enabled.
        failure_probability: The probability of introducing a failure (0.0 to 1.0).
        enabled_failure_modes: The types of failures to introduce. If None, all types are enabled.
        exception_types: The types of exceptions to raise. If None, ValueError is used.
        max_delay_ms: The maximum delay to introduce in milliseconds.
        seed: The random seed to use. If None, a random seed is used.
        
    Yields:
        The chaos configuration.
    """
    # Save the original configuration
    original_config = get_chaos_config()
    
    # Create a new configuration
    config = ChaosConfig(
        enabled=enabled,
        failure_probability=failure_probability,
        enabled_failure_modes=enabled_failure_modes,
        exception_types=exception_types,
        max_delay_ms=max_delay_ms,
        seed=seed,
    )
    
    # Set the new configuration
    set_chaos_config(config)
    
    try:
        # Yield the configuration
        yield config
    finally:
        # Restore the original configuration
        set_chaos_config(original_config)


class ChaosTester:
    """
    Tester for chaos testing.
    
    This class provides methods for testing the resilience of a system
    by introducing various types of failures.
    """
    
    def __init__(
        self,
        failure_probability: float = 0.5,
        enabled_failure_modes: Optional[List[FailureMode]] = None,
        exception_types: Optional[List[Type[Exception]]] = None,
        max_delay_ms: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize a ChaosTester.
        
        Args:
            failure_probability: The probability of introducing a failure (0.0 to 1.0).
            enabled_failure_modes: The types of failures to introduce. If None, all types are enabled.
            exception_types: The types of exceptions to raise. If None, ValueError is used.
            max_delay_ms: The maximum delay to introduce in milliseconds.
            seed: The random seed to use. If None, a random seed is used.
        """
        self.config = ChaosConfig(
            enabled=True,
            failure_probability=failure_probability,
            enabled_failure_modes=enabled_failure_modes,
            exception_types=exception_types,
            max_delay_ms=max_delay_ms,
            seed=seed,
        )
    
    def test_function(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> Optional[R]:
        """
        Test a function with chaos.
        
        Args:
            func: The function to test.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function, or None if an exception was raised.
        """
        with chaos_context(
            enabled=True,
            failure_probability=self.config.failure_probability,
            enabled_failure_modes=self.config.enabled_failure_modes,
            exception_types=self.config.exception_types,
            max_delay_ms=self.config.max_delay_ms,
            seed=self.config.random.randint(0, 1000000),
        ):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.warning(f"Chaos test caught exception: {e}")
                return None
    
    def test_function_multiple(
        self,
        func: Callable[..., R],
        iterations: int,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Test a function multiple times with chaos.
        
        Args:
            func: The function to test.
            iterations: The number of iterations to run.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            A dictionary with the test results.
        """
        results = {
            "iterations": iterations,
            "successes": 0,
            "failures": 0,
            "exceptions": {},
        }
        
        for _ in range(iterations):
            try:
                result = self.test_function(func, *args, **kwargs)
                if result is not None:
                    results["successes"] += 1
                else:
                    results["failures"] += 1
            except Exception as e:
                results["failures"] += 1
                exception_type = type(e).__name__
                results["exceptions"][exception_type] = results["exceptions"].get(exception_type, 0) + 1
        
        return results
