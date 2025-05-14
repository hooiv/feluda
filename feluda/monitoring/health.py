"""
Health checks module for Feluda.

This module provides health checks for Feluda.
"""

import abc
import enum
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class HealthStatus(str, enum.Enum):
    """Enum for health status."""
    
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthCheck(BaseModel):
    """
    Health check.
    
    This class represents a health check.
    """
    
    name: str = Field(..., description="The health check name")
    description: str = Field(..., description="The health check description")
    check_function: Callable[[], HealthStatus] = Field(..., description="The health check function")
    interval: float = Field(60.0, description="The health check interval in seconds")
    timeout: float = Field(10.0, description="The health check timeout in seconds")
    last_check_time: Optional[float] = Field(None, description="The last check time")
    last_status: HealthStatus = Field(HealthStatus.UNKNOWN, description="The last health status")
    last_error: Optional[str] = Field(None, description="The last error")
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the health check to a dictionary.
        
        Returns:
            A dictionary representation of the health check.
        """
        return {
            "name": self.name,
            "description": self.description,
            "interval": self.interval,
            "timeout": self.timeout,
            "last_check_time": self.last_check_time,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }
    
    def check(self) -> HealthStatus:
        """
        Run the health check.
        
        Returns:
            The health status.
        """
        self.last_check_time = time.time()
        
        try:
            # Run the health check with a timeout
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.check_function)
                
                try:
                    status = future.result(timeout=self.timeout)
                    self.last_status = status
                    self.last_error = None
                    
                    return status
                
                except concurrent.futures.TimeoutError:
                    self.last_status = HealthStatus.UNHEALTHY
                    self.last_error = f"Health check timed out after {self.timeout} seconds"
                    
                    return HealthStatus.UNHEALTHY
        
        except Exception as e:
            self.last_status = HealthStatus.UNHEALTHY
            self.last_error = str(e)
            
            return HealthStatus.UNHEALTHY


class HealthCheckManager:
    """
    Health check manager.
    
    This class is responsible for managing health checks.
    """
    
    def __init__(self):
        """
        Initialize the health check manager.
        """
        self.health_checks: Dict[str, HealthCheck] = {}
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            health_check: The health check to register.
        """
        with self.lock:
            self.health_checks[health_check.name] = health_check
    
    def get_health_check(self, name: str) -> Optional[HealthCheck]:
        """
        Get a health check by name.
        
        Args:
            name: The health check name.
            
        Returns:
            The health check, or None if the health check is not found.
        """
        with self.lock:
            return self.health_checks.get(name)
    
    def get_health_checks(self) -> Dict[str, HealthCheck]:
        """
        Get all health checks.
        
        Returns:
            A dictionary mapping health check names to health checks.
        """
        with self.lock:
            return self.health_checks.copy()
    
    def check_health(self, name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """
        Check health.
        
        Args:
            name: The health check name. If None, check all health checks.
            
        Returns:
            A dictionary mapping health check names to health statuses.
        """
        with self.lock:
            if name:
                health_check = self.get_health_check(name)
                
                if not health_check:
                    return {}
                
                return {name: health_check.check()}
            
            return {
                name: health_check.check()
                for name, health_check in self.health_checks.items()
            }
    
    def get_health_status(self, name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """
        Get health status.
        
        Args:
            name: The health check name. If None, get the status of all health checks.
            
        Returns:
            A dictionary mapping health check names to health statuses.
        """
        with self.lock:
            if name:
                health_check = self.get_health_check(name)
                
                if not health_check:
                    return {}
                
                return {name: health_check.last_status}
            
            return {
                name: health_check.last_status
                for name, health_check in self.health_checks.items()
            }
    
    def get_overall_health_status(self) -> HealthStatus:
        """
        Get the overall health status.
        
        Returns:
            The overall health status.
        """
        with self.lock:
            statuses = [
                health_check.last_status
                for health_check in self.health_checks.values()
            ]
            
            if not statuses:
                return HealthStatus.UNKNOWN
            
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            
            if HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            
            if HealthStatus.UNKNOWN in statuses:
                return HealthStatus.UNKNOWN
            
            return HealthStatus.HEALTHY
    
    def start(self) -> None:
        """
        Start the health check manager.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run_health_checks)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the health check manager.
        """
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def _run_health_checks(self) -> None:
        """
        Run health checks periodically.
        """
        while self.running:
            try:
                # Get the health checks
                health_checks = self.get_health_checks()
                
                # Check if any health checks need to be run
                for name, health_check in health_checks.items():
                    if (
                        health_check.last_check_time is None
                        or time.time() - health_check.last_check_time >= health_check.interval
                    ):
                        try:
                            health_check.check()
                        except Exception as e:
                            log.error(f"Error running health check {name}: {e}")
                
                # Sleep for a short time
                time.sleep(1.0)
            
            except Exception as e:
                log.error(f"Error running health checks: {e}")
                time.sleep(1.0)


# Global health check manager instance
_health_check_manager = None
_health_check_manager_lock = threading.RLock()


def get_health_check_manager() -> HealthCheckManager:
    """
    Get the global health check manager instance.
    
    Returns:
        The global health check manager instance.
    """
    global _health_check_manager
    
    with _health_check_manager_lock:
        if _health_check_manager is None:
            _health_check_manager = HealthCheckManager()
            _health_check_manager.start()
        
        return _health_check_manager
