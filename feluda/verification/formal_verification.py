"""
Formal Verification Module

This module provides formal verification capabilities for Feluda.
"""

import inspect
import logging
import sys
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

import deal

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class VerificationResult(str, Enum):
    """Enum for verification results."""
    
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"
    ERROR = "error"


class VerificationReport:
    """
    Report for a verification run.
    
    This class holds the results of a verification run.
    """
    
    def __init__(
        self,
        function_name: str,
        result: VerificationResult,
        execution_time: float,
        counterexample: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ):
        """
        Initialize a VerificationReport.
        
        Args:
            function_name: The name of the verified function.
            result: The verification result.
            execution_time: The execution time in seconds.
            counterexample: A counterexample if the verification failed.
            error_message: An error message if the verification failed.
        """
        self.function_name = function_name
        self.result = result
        self.execution_time = execution_time
        self.counterexample = counterexample
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary.
        
        Returns:
            A dictionary representation of the report.
        """
        return {
            "function_name": self.function_name,
            "result": self.result,
            "execution_time": self.execution_time,
            "counterexample": self.counterexample,
            "error_message": self.error_message,
        }
    
    def __str__(self) -> str:
        """
        Convert the report to a string.
        
        Returns:
            A string representation of the report.
        """
        result = f"Verification of {self.function_name}: {self.result.value}"
        result += f" (execution time: {self.execution_time:.2f}s)"
        
        if self.counterexample:
            result += f"\nCounterexample: {self.counterexample}"
        
        if self.error_message:
            result += f"\nError: {self.error_message}"
        
        return result


class FormalVerifier:
    """
    Base class for formal verifiers.
    
    This class defines the interface for formal verifiers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize a FormalVerifier.
        
        Args:
            timeout: The timeout in seconds.
        """
        self.timeout = timeout
    
    def verify(self, function: Callable[..., Any]) -> VerificationReport:
        """
        Verify a function.
        
        Args:
            function: The function to verify.
            
        Returns:
            A verification report.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement verify")


class DealVerifier(FormalVerifier):
    """
    Formal verifier using the deal library.
    
    This class implements formal verification using the deal library.
    """
    
    def verify(self, function: Callable[..., Any]) -> VerificationReport:
        """
        Verify a function using the deal library.
        
        Args:
            function: The function to verify.
            
        Returns:
            A verification report.
        """
        function_name = function.__name__
        start_time = time.time()
        
        # Check if the function has contracts
        pre_conditions = getattr(function, "__deal_pre__", [])
        post_conditions = getattr(function, "__deal_post__", [])
        invariants = getattr(function, "__deal_inv__", [])
        
        if not pre_conditions and not post_conditions and not invariants:
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.UNKNOWN,
                execution_time=time.time() - start_time,
                error_message="No contracts found",
            )
        
        try:
            # Run the deal validator
            validator = deal.cases(function)
            validator.run(count=100, timeout=self.timeout)
            
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.VERIFIED,
                execution_time=time.time() - start_time,
            )
        except deal.cases.TestFailed as e:
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.FALSIFIED,
                execution_time=time.time() - start_time,
                counterexample=e.case,
                error_message=str(e),
            )
        except Exception as e:
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.ERROR,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )


class CrosshairVerifier(FormalVerifier):
    """
    Formal verifier using the CrossHair library.
    
    This class implements formal verification using the CrossHair library.
    """
    
    def verify(self, function: Callable[..., Any]) -> VerificationReport:
        """
        Verify a function using the CrossHair library.
        
        Args:
            function: The function to verify.
            
        Returns:
            A verification report.
        """
        function_name = function.__name__
        start_time = time.time()
        
        try:
            # Try to import CrossHair
            from crosshair.core import analyze_function
            from crosshair.options import AnalysisOptions
            from crosshair.condition_parser import ConditionParser
            from crosshair.statespace import RootState
            from crosshair.util import set_debug
        except ImportError:
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.ERROR,
                execution_time=time.time() - start_time,
                error_message="CrossHair not installed",
            )
        
        try:
            # Run CrossHair
            options = AnalysisOptions(
                timeout_seconds=self.timeout,
                max_iterations=1000,
            )
            
            results = list(analyze_function(function, options))
            
            if not results:
                return VerificationReport(
                    function_name=function_name,
                    result=VerificationResult.UNKNOWN,
                    execution_time=time.time() - start_time,
                    error_message="No results from CrossHair",
                )
            
            # Check if any verification failed
            for result in results:
                if result.error:
                    return VerificationReport(
                        function_name=function_name,
                        result=VerificationResult.FALSIFIED,
                        execution_time=time.time() - start_time,
                        counterexample={"args": str(result.args)},
                        error_message=result.error,
                    )
            
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.VERIFIED,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return VerificationReport(
                function_name=function_name,
                result=VerificationResult.ERROR,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )


def create_verifier(verifier_type: str = "deal", timeout: float = 10.0) -> FormalVerifier:
    """
    Create a formal verifier.
    
    Args:
        verifier_type: The type of verifier to create.
        timeout: The timeout in seconds.
        
    Returns:
        A formal verifier.
        
    Raises:
        ValueError: If the verifier type is not supported.
    """
    if verifier_type == "deal":
        return DealVerifier(timeout=timeout)
    elif verifier_type == "crosshair":
        return CrosshairVerifier(timeout=timeout)
    else:
        raise ValueError(f"Unsupported verifier type: {verifier_type}")


def verify_function(
    function: Callable[..., Any],
    verifier_type: str = "deal",
    timeout: float = 10.0,
) -> VerificationReport:
    """
    Verify a function.
    
    Args:
        function: The function to verify.
        verifier_type: The type of verifier to use.
        timeout: The timeout in seconds.
        
    Returns:
        A verification report.
    """
    verifier = create_verifier(verifier_type=verifier_type, timeout=timeout)
    return verifier.verify(function)


def verify_module(
    module: Any,
    verifier_type: str = "deal",
    timeout: float = 10.0,
) -> List[VerificationReport]:
    """
    Verify all functions in a module.
    
    Args:
        module: The module to verify.
        verifier_type: The type of verifier to use.
        timeout: The timeout in seconds.
        
    Returns:
        A list of verification reports.
    """
    verifier = create_verifier(verifier_type=verifier_type, timeout=timeout)
    reports = []
    
    # Find all functions in the module
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            report = verifier.verify(obj)
            reports.append(report)
    
    return reports


def verify_class(
    cls: Type,
    verifier_type: str = "deal",
    timeout: float = 10.0,
) -> List[VerificationReport]:
    """
    Verify all methods in a class.
    
    Args:
        cls: The class to verify.
        verifier_type: The type of verifier to use.
        timeout: The timeout in seconds.
        
    Returns:
        A list of verification reports.
    """
    verifier = create_verifier(verifier_type=verifier_type, timeout=timeout)
    reports = []
    
    # Find all methods in the class
    for name, obj in inspect.getmembers(cls):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            report = verifier.verify(obj)
            reports.append(report)
    
    return reports


@deal.pre(lambda f: callable(f))
@deal.post(lambda result: isinstance(result, bool))
def has_contracts(f: Callable[..., Any]) -> bool:
    """
    Check if a function has contracts.
    
    Args:
        f: The function to check.
        
    Returns:
        True if the function has contracts, False otherwise.
    """
    pre_conditions = getattr(f, "__deal_pre__", [])
    post_conditions = getattr(f, "__deal_post__", [])
    invariants = getattr(f, "__deal_inv__", [])
    
    return bool(pre_conditions or post_conditions or invariants)
