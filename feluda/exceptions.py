"""
Feluda Exception Hierarchy

This module defines a comprehensive exception hierarchy for the Feluda framework.
All Feluda-specific exceptions inherit from FeludaError, which itself inherits from Exception.
This allows for more granular error handling and better error messages.
"""

from typing import Any, Dict, List, Optional, Union


class FeludaError(Exception):
    """Base exception class for all Feluda-specific errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a FeludaError.

        Args:
            message: A human-readable error message.
            details: Optional dictionary with additional error details.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


# Configuration Errors

class ConfigurationError(FeludaError):
    """Base class for configuration-related errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when a configuration is invalid."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when a required configuration is missing."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, validation_errors: List[Dict[str, Any]]):
        """
        Initialize a ConfigurationValidationError.
        
        Args:
            message: A human-readable error message.
            validation_errors: List of validation errors.
        """
        super().__init__(message, {"validation_errors": validation_errors})
        self.validation_errors = validation_errors


# Operator Errors

class OperatorError(FeludaError):
    """Base class for operator-related errors."""
    
    def __init__(self, message: str, operator_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize an OperatorError.
        
        Args:
            message: A human-readable error message.
            operator_type: The type of operator that caused the error.
            details: Optional dictionary with additional error details.
        """
        error_details = details or {}
        if operator_type:
            error_details["operator_type"] = operator_type
        super().__init__(message, error_details)
        self.operator_type = operator_type


class OperatorInitializationError(OperatorError):
    """Raised when an operator fails to initialize."""
    pass


class OperatorNotFoundError(OperatorError):
    """Raised when an operator is not found."""
    pass


class OperatorExecutionError(OperatorError):
    """Raised when an operator fails during execution."""
    
    def __init__(
        self, 
        message: str, 
        operator_type: Optional[str] = None, 
        input_data: Optional[Any] = None,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an OperatorExecutionError.
        
        Args:
            message: A human-readable error message.
            operator_type: The type of operator that caused the error.
            input_data: The input data that caused the error.
            cause: The original exception that caused this error.
            details: Optional dictionary with additional error details.
        """
        error_details = details or {}
        if input_data is not None:
            error_details["input_data"] = str(input_data)
        if cause:
            error_details["cause"] = str(cause)
        super().__init__(message, operator_type, error_details)
        self.input_data = input_data
        self.cause = cause


class OperatorValidationError(OperatorError):
    """Raised when operator validation fails."""
    pass


class OperatorContractError(OperatorError):
    """Raised when an operator contract is violated."""
    
    def __init__(
        self, 
        message: str, 
        operator_type: Optional[str] = None, 
        contract_type: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an OperatorContractError.
        
        Args:
            message: A human-readable error message.
            operator_type: The type of operator that caused the error.
            contract_type: The type of contract that was violated (pre, post, inv).
            details: Optional dictionary with additional error details.
        """
        error_details = details or {}
        error_details["contract_type"] = contract_type
        super().__init__(message, operator_type, error_details)
        self.contract_type = contract_type


# Resource Errors

class ResourceError(FeludaError):
    """Base class for resource-related errors."""
    pass


class ResourceNotFoundError(ResourceError):
    """Raised when a resource is not found."""
    pass


class ResourceAccessError(ResourceError):
    """Raised when a resource cannot be accessed."""
    pass


# Data Errors

class DataError(FeludaError):
    """Base class for data-related errors."""
    pass


class InvalidDataError(DataError):
    """Raised when data is invalid."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_errors: List[Dict[str, Any]]):
        """
        Initialize a DataValidationError.
        
        Args:
            message: A human-readable error message.
            validation_errors: List of validation errors.
        """
        super().__init__(message, {"validation_errors": validation_errors})
        self.validation_errors = validation_errors


# Security Errors

class SecurityError(FeludaError):
    """Base class for security-related errors."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


# System Errors

class SystemError(FeludaError):
    """Base class for system-related errors."""
    pass


class HardwareError(SystemError):
    """Raised when a hardware-related error occurs."""
    pass


class PerformanceError(SystemError):
    """Raised when a performance-related error occurs."""
    pass


class CircuitBreakerError(SystemError):
    """Raised when a circuit breaker is triggered."""
    
    def __init__(
        self, 
        message: str, 
        service_name: str,
        failure_count: int,
        threshold: int,
        reset_timeout: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a CircuitBreakerError.
        
        Args:
            message: A human-readable error message.
            service_name: The name of the service that triggered the circuit breaker.
            failure_count: The current failure count.
            threshold: The failure threshold that triggered the circuit breaker.
            reset_timeout: The timeout in seconds before the circuit breaker resets.
            details: Optional dictionary with additional error details.
        """
        error_details = details or {}
        error_details.update({
            "service_name": service_name,
            "failure_count": failure_count,
            "threshold": threshold,
            "reset_timeout": reset_timeout
        })
        super().__init__(message, error_details)
        self.service_name = service_name
        self.failure_count = failure_count
        self.threshold = threshold
        self.reset_timeout = reset_timeout
