"""
Circuit Breaker Module

This module implements the Circuit Breaker pattern for resilience.
The Circuit Breaker pattern prevents an application from repeatedly trying to execute
an operation that's likely to fail, allowing it to continue without waiting for the
fault to be fixed or wasting resources while the fault is being fixed.
"""

import enum
import logging
import time
from functools import wraps
from threading import Lock, RLock
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import deal

from feluda.exceptions import CircuitBreakerError

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class CircuitState(enum.Enum):
    """Enum representing the state of a circuit breaker."""
    
    CLOSED = "closed"  # Circuit is closed, requests are allowed through
    OPEN = "open"  # Circuit is open, requests are not allowed through
    HALF_OPEN = "half_open"  # Circuit is half-open, limited requests are allowed through


class CircuitBreaker(Generic[T, R]):
    """
    Implementation of the Circuit Breaker pattern.
    
    The Circuit Breaker pattern prevents an application from repeatedly trying to execute
    an operation that's likely to fail, allowing it to continue without waiting for the
    fault to be fixed or wasting resources while the fault is being fixed.
    
    Type Parameters:
        T: The type of the function being wrapped.
        R: The return type of the function being wrapped.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: Optional[List[Type[Exception]]] = None,
        fallback_function: Optional[Callable[..., R]] = None,
    ):
        """
        Initialize a CircuitBreaker.
        
        Args:
            name: The name of the circuit breaker.
            failure_threshold: The number of consecutive failures required to open the circuit.
            recovery_timeout: The time in seconds to wait before transitioning from OPEN to HALF_OPEN.
            expected_exceptions: A list of exception types that should be counted as failures.
                                If None, all exceptions are counted as failures.
            fallback_function: A function to call when the circuit is open.
                              If None, a CircuitBreakerError is raised.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions or [Exception]
        self.fallback_function = fallback_function
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state_lock = RLock()
        
        # Registry of all circuit breakers
        CircuitBreakerRegistry.register(self)
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get the current failure count."""
        return self._failure_count
    
    @property
    def last_failure_time(self) -> float:
        """Get the timestamp of the last failure."""
        return self._last_failure_time
    
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """
        Decorate a function with the circuit breaker.
        
        Args:
            func: The function to decorate.
            
        Returns:
            The decorated function.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return self.call(func, *args, **kwargs)
        
        return wrapper
    
    @deal.raises(CircuitBreakerError)
    def call(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """
        Call the function with circuit breaker protection.
        
        Args:
            func: The function to call.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function call.
            
        Raises:
            CircuitBreakerError: If the circuit is open and no fallback function is provided.
        """
        with self._state_lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    log.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                else:
                    return self._handle_open_circuit(func, *args, **kwargs)
        
        try:
            result = func(*args, **kwargs)
            
            with self._state_lock:
                if self._state == CircuitState.HALF_OPEN:
                    log.info(f"Circuit {self.name} transitioning from HALF_OPEN to CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            
            return result
            
        except Exception as e:
            return self._handle_exception(e, func, *args, **kwargs)
    
    def _handle_exception(self, e: Exception, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """
        Handle an exception that occurred during a function call.
        
        Args:
            e: The exception that occurred.
            func: The function that was called.
            *args: Positional arguments that were passed to the function.
            **kwargs: Keyword arguments that were passed to the function.
            
        Returns:
            The result of the fallback function if provided.
            
        Raises:
            CircuitBreakerError: If no fallback function is provided.
            Exception: If the exception is not one of the expected exceptions.
        """
        if not any(isinstance(e, exc_type) for exc_type in self.expected_exceptions):
            # If the exception is not one of the expected exceptions, re-raise it
            raise
        
        with self._state_lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                log.warning(f"Circuit {self.name} transitioning from CLOSED to OPEN")
                self._state = CircuitState.OPEN
            
            if self._state == CircuitState.HALF_OPEN:
                log.warning(f"Circuit {self.name} transitioning from HALF_OPEN to OPEN")
                self._state = CircuitState.OPEN
        
        return self._handle_open_circuit(func, *args, **kwargs)
    
    def _handle_open_circuit(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """
        Handle a call when the circuit is open.
        
        Args:
            func: The function that was called.
            *args: Positional arguments that were passed to the function.
            **kwargs: Keyword arguments that were passed to the function.
            
        Returns:
            The result of the fallback function if provided.
            
        Raises:
            CircuitBreakerError: If no fallback function is provided.
        """
        if self.fallback_function:
            return self.fallback_function(*args, **kwargs)
        
        raise CircuitBreakerError(
            message=f"Circuit {self.name} is OPEN",
            service_name=self.name,
            failure_count=self._failure_count,
            threshold=self.failure_threshold,
            reset_timeout=self.recovery_timeout
        )
    
    def reset(self) -> None:
        """Reset the circuit breaker to its initial state."""
        with self._state_lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            log.info(f"Circuit {self.name} has been reset")


class CircuitBreakerRegistry:
    """
    Registry of all circuit breakers.
    
    This class provides a way to access all circuit breakers in the application.
    """
    
    _registry: Dict[str, CircuitBreaker] = {}
    _lock = Lock()
    
    @classmethod
    def register(cls, circuit_breaker: CircuitBreaker) -> None:
        """
        Register a circuit breaker.
        
        Args:
            circuit_breaker: The circuit breaker to register.
            
        Raises:
            ValueError: If a circuit breaker with the same name is already registered.
        """
        with cls._lock:
            if circuit_breaker.name in cls._registry:
                raise ValueError(f"Circuit breaker with name '{circuit_breaker.name}' already exists")
            
            cls._registry[circuit_breaker.name] = circuit_breaker
    
    @classmethod
    def get(cls, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.
        
        Args:
            name: The name of the circuit breaker.
            
        Returns:
            The circuit breaker, or None if not found.
        """
        return cls._registry.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, CircuitBreaker]:
        """
        Get all registered circuit breakers.
        
        Returns:
            A dictionary of all registered circuit breakers, keyed by name.
        """
        return cls._registry.copy()
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all registered circuit breakers."""
        for circuit_breaker in cls._registry.values():
            circuit_breaker.reset()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exceptions: Optional[List[Type[Exception]]] = None,
    fallback_function: Optional[Callable[..., Any]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator for applying a circuit breaker to a function.
    
    Args:
        name: The name of the circuit breaker.
        failure_threshold: The number of consecutive failures required to open the circuit.
        recovery_timeout: The time in seconds to wait before transitioning from OPEN to HALF_OPEN.
        expected_exceptions: A list of exception types that should be counted as failures.
                            If None, all exceptions are counted as failures.
        fallback_function: A function to call when the circuit is open.
                          If None, a CircuitBreakerError is raised.
        
    Returns:
        A decorator that applies a circuit breaker to a function.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        cb = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
            fallback_function=fallback_function,
        )
        return cb(func)
    
    return decorator
