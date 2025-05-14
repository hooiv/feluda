#!/usr/bin/env python
"""
Formal Verification Script for Feluda Operators

This script performs formal verification on Feluda operators using CrossHair.
It verifies that the operators satisfy their contracts and specifications.
"""

import argparse
import importlib
import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Type

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feluda.base_operator import BaseFeludaOperator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("verify_operators")

# Try to import CrossHair, but don't fail if it's not available
try:
    import crosshair
    from crosshair.core import analyze_function
    from crosshair.options import AnalysisOptions
    CROSSHAIR_AVAILABLE = True
except ImportError:
    CROSSHAIR_AVAILABLE = False
    log.warning("CrossHair not available. Install it with 'pip install crosshair-tool'.")


def discover_operators(operators_dir: str = "operators") -> List[Tuple[str, Type[BaseFeludaOperator]]]:
    """
    Discover all operator classes in the operators directory.
    
    Args:
        operators_dir: The directory containing the operators.
        
    Returns:
        A list of tuples containing the module name and operator class.
    """
    operators = []
    
    # Get the absolute path to the operators directory
    operators_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", operators_dir))
    
    # Iterate over all subdirectories in the operators directory
    for operator_dir in os.listdir(operators_path):
        operator_path = os.path.join(operators_path, operator_dir)
        
        # Skip files and hidden directories
        if not os.path.isdir(operator_path) or operator_dir.startswith("."):
            continue
        
        # Check if the operator module exists
        module_path = os.path.join(operator_path, f"{operator_dir}.py")
        if not os.path.isfile(module_path):
            continue
        
        try:
            # Import the module
            module_name = f"operators.{operator_dir}.{operator_dir}"
            module = importlib.import_module(module_name)
            
            # Find all BaseFeludaOperator subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseFeludaOperator)
                    and obj != BaseFeludaOperator
                ):
                    operators.append((module_name, obj))
        except (ImportError, AttributeError) as e:
            log.warning(f"Failed to import operator {operator_dir}: {e}")
    
    return operators


def verify_operator_with_crosshair(
    operator_class: Type[BaseFeludaOperator],
    timeout_seconds: int = 30,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Verify an operator using CrossHair.
    
    Args:
        operator_class: The operator class to verify.
        timeout_seconds: The maximum time to spend on verification.
        max_iterations: The maximum number of iterations to perform.
        
    Returns:
        A dictionary containing the verification results.
    """
    if not CROSSHAIR_AVAILABLE:
        return {
            "success": False,
            "error": "CrossHair not available",
            "operator": operator_class.__name__
        }
    
    results = {
        "operator": operator_class.__name__,
        "methods": {},
        "success": True
    }
    
    # Create an instance of the operator for verification
    try:
        operator = operator_class()
    except Exception as e:
        log.warning(f"Failed to create instance of {operator_class.__name__}: {e}")
        return {
            "success": False,
            "error": f"Failed to create instance: {str(e)}",
            "operator": operator_class.__name__
        }
    
    # Verify the run method
    options = AnalysisOptions(
        timeout_seconds=timeout_seconds,
        max_iterations=max_iterations
    )
    
    try:
        # Analyze the run method
        run_method = operator.run
        run_results = analyze_function(run_method, options)
        
        # Process the results
        method_results = {
            "success": True,
            "errors": []
        }
        
        for result in run_results:
            if result.error:
                method_results["success"] = False
                method_results["errors"].append({
                    "message": result.error,
                    "condition": result.condition,
                    "args": str(result.args)
                })
        
        results["methods"]["run"] = method_results
        
        if not method_results["success"]:
            results["success"] = False
    
    except Exception as e:
        log.warning(f"Failed to verify {operator_class.__name__}.run: {e}")
        results["methods"]["run"] = {
            "success": False,
            "error": str(e)
        }
        results["success"] = False
    
    # Verify the _execute method
    try:
        # Analyze the _execute method
        execute_method = operator._execute
        execute_results = analyze_function(execute_method, options)
        
        # Process the results
        method_results = {
            "success": True,
            "errors": []
        }
        
        for result in execute_results:
            if result.error:
                method_results["success"] = False
                method_results["errors"].append({
                    "message": result.error,
                    "condition": result.condition,
                    "args": str(result.args)
                })
        
        results["methods"]["_execute"] = method_results
        
        if not method_results["success"]:
            results["success"] = False
    
    except Exception as e:
        log.warning(f"Failed to verify {operator_class.__name__}._execute: {e}")
        results["methods"]["_execute"] = {
            "success": False,
            "error": str(e)
        }
        results["success"] = False
    
    return results


def verify_operator_contracts(operator_class: Type[BaseFeludaOperator]) -> Dict[str, Any]:
    """
    Verify that an operator has the required contracts.
    
    Args:
        operator_class: The operator class to verify.
        
    Returns:
        A dictionary containing the verification results.
    """
    results = {
        "operator": operator_class.__name__,
        "contracts": {},
        "success": True
    }
    
    # Check if the run method has pre and post conditions
    run_method = operator_class.run
    run_contracts = getattr(run_method, "__deal_pre__", []) + getattr(run_method, "__deal_post__", [])
    
    results["contracts"]["run"] = {
        "success": len(run_contracts) > 0,
        "count": len(run_contracts)
    }
    
    if not results["contracts"]["run"]["success"]:
        results["success"] = False
    
    # Check if the _execute method has contracts
    execute_method = operator_class._execute
    execute_contracts = getattr(execute_method, "__deal_pre__", []) + getattr(execute_method, "__deal_post__", [])
    
    results["contracts"]["_execute"] = {
        "success": len(execute_contracts) > 0,
        "count": len(execute_contracts)
    }
    
    if not results["contracts"]["_execute"]["success"]:
        results["success"] = False
    
    return results


def verify_operators(
    operators_dir: str = "operators",
    timeout_seconds: int = 30,
    max_iterations: int = 100,
    verify_with_crosshair: bool = True
) -> Dict[str, Any]:
    """
    Verify all operators in the operators directory.
    
    Args:
        operators_dir: The directory containing the operators.
        timeout_seconds: The maximum time to spend on verification.
        max_iterations: The maximum number of iterations to perform.
        verify_with_crosshair: Whether to verify with CrossHair.
        
    Returns:
        A dictionary containing the verification results.
    """
    results = {
        "operators": [],
        "success": True,
        "total": 0,
        "passed": 0,
        "failed": 0
    }
    
    # Discover operators
    operators = discover_operators(operators_dir)
    results["total"] = len(operators)
    
    # Verify each operator
    for module_name, operator_class in operators:
        log.info(f"Verifying operator {operator_class.__name__} from {module_name}")
        
        operator_results = {
            "name": operator_class.__name__,
            "module": module_name,
            "contract_verification": verify_operator_contracts(operator_class),
        }
        
        if verify_with_crosshair and CROSSHAIR_AVAILABLE:
            operator_results["crosshair_verification"] = verify_operator_with_crosshair(
                operator_class,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations
            )
        
        # Check if the operator passed all verifications
        operator_success = operator_results["contract_verification"]["success"]
        if verify_with_crosshair and CROSSHAIR_AVAILABLE:
            operator_success = operator_success and operator_results["crosshair_verification"]["success"]
        
        operator_results["success"] = operator_success
        
        if operator_success:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["success"] = False
        
        results["operators"].append(operator_results)
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify Feluda operators")
    parser.add_argument(
        "--operators-dir",
        type=str,
        default="operators",
        help="Directory containing the operators"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Maximum time to spend on verification (seconds)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of iterations to perform"
    )
    parser.add_argument(
        "--no-crosshair",
        action="store_true",
        help="Skip verification with CrossHair"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify operators
    results = verify_operators(
        operators_dir=args.operators_dir,
        timeout_seconds=args.timeout,
        max_iterations=args.max_iterations,
        verify_with_crosshair=not args.no_crosshair
    )
    
    # Print results
    print(f"\nVerification Results:")
    print(f"Total operators: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Overall success: {results['success']}")
    
    if results["failed"] > 0:
        print("\nFailed operators:")
        for operator in results["operators"]:
            if not operator["success"]:
                print(f"  - {operator['name']} ({operator['module']})")
    
    # Return exit code based on success
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
