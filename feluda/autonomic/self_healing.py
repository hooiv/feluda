"""
Self-Healing Module

This module provides self-healing capabilities for Feluda.
"""

import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

from feluda.observability import get_logger
from feluda.resilience.circuit_breaker import CircuitBreaker, CircuitState

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class HealingStrategy(str, Enum):
    """Enum for healing strategies."""
    
    RETRY = "retry"
    FALLBACK = "fallback"
    THROTTLE = "throttle"
    SHED_LOAD = "shed_load"
    RECONFIGURE = "reconfigure"
    RESTART = "restart"
    ISOLATE = "isolate"


class HealingAction:
    """
    Action to take for healing.
    
    This class represents an action that can be taken to heal a system.
    """
    
    def __init__(
        self,
        strategy: HealingStrategy,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a HealingAction.
        
        Args:
            strategy: The healing strategy to use.
            params: Additional parameters for the strategy.
        """
        self.strategy = strategy
        self.params = params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the action to a dictionary.
        
        Returns:
            A dictionary representation of the action.
        """
        return {
            "strategy": self.strategy,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealingAction":
        """
        Create an action from a dictionary.
        
        Args:
            data: The dictionary representation of the action.
            
        Returns:
            The created action.
        """
        return cls(
            strategy=data["strategy"],
            params=data.get("params", {}),
        )


class HealthStatus(str, Enum):
    """Enum for health status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck:
    """
    Health check for a component.
    
    This class represents a health check that can be performed on a component.
    """
    
    def __init__(
        self,
        name: str,
        check_function: Callable[[], HealthStatus],
        healing_actions: Dict[HealthStatus, List[HealingAction]],
        check_interval: float = 60.0,
    ):
        """
        Initialize a HealthCheck.
        
        Args:
            name: The name of the health check.
            check_function: The function to call to perform the health check.
            healing_actions: The actions to take for each health status.
            check_interval: The interval in seconds between health checks.
        """
        self.name = name
        self.check_function = check_function
        self.healing_actions = healing_actions
        self.check_interval = check_interval
        self.last_check_time = 0.0
        self.last_status = HealthStatus.UNKNOWN
    
    def should_check(self) -> bool:
        """
        Check if the health check should be performed.
        
        Returns:
            True if the health check should be performed, False otherwise.
        """
        return time.time() - self.last_check_time >= self.check_interval
    
    def check(self) -> HealthStatus:
        """
        Perform the health check.
        
        Returns:
            The health status.
        """
        self.last_check_time = time.time()
        self.last_status = self.check_function()
        return self.last_status
    
    def get_healing_actions(self) -> List[HealingAction]:
        """
        Get the healing actions for the current health status.
        
        Returns:
            The healing actions.
        """
        return self.healing_actions.get(self.last_status, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the health check to a dictionary.
        
        Returns:
            A dictionary representation of the health check.
        """
        return {
            "name": self.name,
            "healing_actions": {
                status.value: [action.to_dict() for action in actions]
                for status, actions in self.healing_actions.items()
            },
            "check_interval": self.check_interval,
            "last_check_time": self.last_check_time,
            "last_status": self.last_status,
        }


class SelfHealingSystem:
    """
    Self-healing system.
    
    This class manages health checks and healing actions for a system.
    """
    
    def __init__(self):
        """Initialize a SelfHealingSystem."""
        self.health_checks: Dict[str, HealthCheck] = {}
        self.healing_history: List[Dict[str, Any]] = []
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """
        Add a health check.
        
        Args:
            health_check: The health check to add.
        """
        self.health_checks[health_check.name] = health_check
    
    def remove_health_check(self, name: str) -> None:
        """
        Remove a health check.
        
        Args:
            name: The name of the health check to remove.
        """
        if name in self.health_checks:
            del self.health_checks[name]
    
    def check_health(self, name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """
        Check the health of the system.
        
        Args:
            name: The name of the health check to perform. If None, all health checks are performed.
            
        Returns:
            A dictionary mapping health check names to health statuses.
        """
        results = {}
        
        if name is not None:
            if name in self.health_checks:
                health_check = self.health_checks[name]
                if health_check.should_check():
                    results[name] = health_check.check()
                else:
                    results[name] = health_check.last_status
        else:
            for name, health_check in self.health_checks.items():
                if health_check.should_check():
                    results[name] = health_check.check()
                else:
                    results[name] = health_check.last_status
        
        return results
    
    def heal(self, name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Heal the system.
        
        Args:
            name: The name of the health check to heal. If None, all health checks are healed.
            
        Returns:
            A dictionary mapping health check names to lists of healing actions taken.
        """
        results = {}
        
        if name is not None:
            if name in self.health_checks:
                health_check = self.health_checks[name]
                results[name] = self._heal_component(health_check)
        else:
            for name, health_check in self.health_checks.items():
                results[name] = self._heal_component(health_check)
        
        return results
    
    def _heal_component(self, health_check: HealthCheck) -> List[Dict[str, Any]]:
        """
        Heal a component.
        
        Args:
            health_check: The health check for the component.
            
        Returns:
            A list of healing actions taken.
        """
        actions = []
        
        # Get the healing actions for the current health status
        healing_actions = health_check.get_healing_actions()
        
        for action in healing_actions:
            # Apply the healing action
            result = self._apply_healing_action(health_check, action)
            
            # Record the action
            action_record = {
                "time": time.time(),
                "health_check": health_check.name,
                "status": health_check.last_status,
                "action": action.to_dict(),
                "result": result,
            }
            
            self.healing_history.append(action_record)
            actions.append(action_record)
        
        return actions
    
    def _apply_healing_action(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply a healing action.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        if action.strategy == HealingStrategy.RETRY:
            return self._apply_retry_strategy(health_check, action)
        elif action.strategy == HealingStrategy.FALLBACK:
            return self._apply_fallback_strategy(health_check, action)
        elif action.strategy == HealingStrategy.THROTTLE:
            return self._apply_throttle_strategy(health_check, action)
        elif action.strategy == HealingStrategy.SHED_LOAD:
            return self._apply_shed_load_strategy(health_check, action)
        elif action.strategy == HealingStrategy.RECONFIGURE:
            return self._apply_reconfigure_strategy(health_check, action)
        elif action.strategy == HealingStrategy.RESTART:
            return self._apply_restart_strategy(health_check, action)
        elif action.strategy == HealingStrategy.ISOLATE:
            return self._apply_isolate_strategy(health_check, action)
        else:
            return {"success": False, "error": f"Unknown healing strategy: {action.strategy}"}
    
    def _apply_retry_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the retry strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the retry parameters
        max_retries = action.params.get("max_retries", 3)
        retry_delay = action.params.get("retry_delay", 1.0)
        
        # Try to recover the component
        for i in range(max_retries):
            log.info(f"Retrying health check {health_check.name} (attempt {i+1}/{max_retries})")
            
            # Perform the health check
            status = health_check.check()
            
            if status == HealthStatus.HEALTHY:
                return {"success": True, "retries": i+1}
            
            # Wait before retrying
            time.sleep(retry_delay)
        
        return {"success": False, "retries": max_retries}
    
    def _apply_fallback_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the fallback strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the fallback parameters
        fallback_function = action.params.get("fallback_function")
        
        if fallback_function is None:
            return {"success": False, "error": "No fallback function specified"}
        
        try:
            # Call the fallback function
            fallback_function()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _apply_throttle_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the throttle strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the throttle parameters
        rate_limit = action.params.get("rate_limit", 10)
        
        # Apply the rate limit
        # This is a placeholder implementation
        
        return {"success": True, "rate_limit": rate_limit}
    
    def _apply_shed_load_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the shed load strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the shed load parameters
        load_reduction = action.params.get("load_reduction", 0.5)
        
        # Shed load
        # This is a placeholder implementation
        
        return {"success": True, "load_reduction": load_reduction}
    
    def _apply_reconfigure_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the reconfigure strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the reconfigure parameters
        config_changes = action.params.get("config_changes", {})
        
        # Apply the configuration changes
        # This is a placeholder implementation
        
        return {"success": True, "config_changes": config_changes}
    
    def _apply_restart_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the restart strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the restart parameters
        restart_function = action.params.get("restart_function")
        
        if restart_function is None:
            return {"success": False, "error": "No restart function specified"}
        
        try:
            # Call the restart function
            restart_function()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _apply_isolate_strategy(self, health_check: HealthCheck, action: HealingAction) -> Dict[str, Any]:
        """
        Apply the isolate strategy.
        
        Args:
            health_check: The health check for the component.
            action: The healing action to apply.
            
        Returns:
            The result of the healing action.
        """
        # Get the isolate parameters
        circuit_breaker_name = action.params.get("circuit_breaker_name")
        
        if circuit_breaker_name is None:
            return {"success": False, "error": "No circuit breaker name specified"}
        
        try:
            from feluda.resilience.circuit_breaker import CircuitBreakerRegistry
            
            # Get the circuit breaker
            circuit_breaker = CircuitBreakerRegistry.get(circuit_breaker_name)
            
            if circuit_breaker is None:
                return {"success": False, "error": f"Circuit breaker not found: {circuit_breaker_name}"}
            
            # Open the circuit breaker
            with circuit_breaker._state_lock:
                circuit_breaker._state = CircuitState.OPEN
                circuit_breaker._failure_count = circuit_breaker.failure_threshold
                circuit_breaker._last_failure_time = time.time()
            
            return {"success": True, "circuit_breaker": circuit_breaker_name}
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_circuit_breaker_health_check(
    name: str,
    circuit_breaker_name: str,
    check_interval: float = 60.0,
) -> HealthCheck:
    """
    Create a health check for a circuit breaker.
    
    Args:
        name: The name of the health check.
        circuit_breaker_name: The name of the circuit breaker.
        check_interval: The interval in seconds between health checks.
        
    Returns:
        A health check for the circuit breaker.
    """
    from feluda.resilience.circuit_breaker import CircuitBreakerRegistry, CircuitState
    
    def check_function() -> HealthStatus:
        circuit_breaker = CircuitBreakerRegistry.get(circuit_breaker_name)
        
        if circuit_breaker is None:
            return HealthStatus.UNKNOWN
        
        if circuit_breaker.state == CircuitState.CLOSED:
            return HealthStatus.HEALTHY
        elif circuit_breaker.state == CircuitState.HALF_OPEN:
            return HealthStatus.DEGRADED
        elif circuit_breaker.state == CircuitState.OPEN:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    healing_actions = {
        HealthStatus.DEGRADED: [
            HealingAction(
                strategy=HealingStrategy.RETRY,
                params={"max_retries": 3, "retry_delay": 1.0},
            ),
        ],
        HealthStatus.UNHEALTHY: [
            HealingAction(
                strategy=HealingStrategy.FALLBACK,
                params={"fallback_function": lambda: None},
            ),
            HealingAction(
                strategy=HealingStrategy.RECONFIGURE,
                params={"config_changes": {"recovery_timeout": 30.0}},
            ),
        ],
    }
    
    return HealthCheck(
        name=name,
        check_function=check_function,
        healing_actions=healing_actions,
        check_interval=check_interval,
    )
