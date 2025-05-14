#!/usr/bin/env python
"""
Script to run formal verification on the vector operations module.

This script uses CrossHair to verify the contracts in the vector operations module.
"""

import argparse
import os
import sys
from typing import List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_crosshair(module_path: str, timeout: int = 30, verbose: bool = False) -> int:
    """
    Run CrossHair on the specified module.
    
    Args:
        module_path: The path to the module to verify.
        timeout: The timeout in seconds for each function verification.
        verbose: Whether to print verbose output.
        
    Returns:
        The exit code (0 for success, non-zero for failure).
    """
    import crosshair
    
    args = ["check", module_path, "--timeout", str(timeout)]
    if verbose:
        args.append("--verbose")
    
    print(f"Running CrossHair on {module_path} with timeout {timeout}s")
    return crosshair.main(args)


def run_deal(module_path: str, verbose: bool = False) -> int:
    """
    Run deal.test on the specified module.
    
    Args:
        module_path: The path to the module to verify.
        verbose: Whether to print verbose output.
        
    Returns:
        The exit code (0 for success, non-zero for failure).
    """
    import deal.linter
    
    print(f"Running deal.linter on {module_path}")
    result = deal.linter.lint([module_path])
    
    if verbose:
        for error in result:
            print(f"{error.path}:{error.line}: {error.message}")
    
    return 1 if result else 0


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point.
    
    Args:
        args: Command line arguments.
        
    Returns:
        The exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Run formal verification on the vector operations module.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for each function verification.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("--tool", choices=["crosshair", "deal", "both"], default="both", help="The verification tool to use.")
    
    parsed_args = parser.parse_args(args)
    
    module_path = "feluda.verification.vector_operations"
    exit_code = 0
    
    if parsed_args.tool in ["crosshair", "both"]:
        try:
            crosshair_exit_code = run_crosshair(module_path, parsed_args.timeout, parsed_args.verbose)
            exit_code = exit_code or crosshair_exit_code
        except ImportError:
            print("CrossHair is not installed. Install it with 'pip install crosshair-tool'.")
            exit_code = 1
    
    if parsed_args.tool in ["deal", "both"]:
        try:
            deal_exit_code = run_deal(module_path, parsed_args.verbose)
            exit_code = exit_code or deal_exit_code
        except ImportError:
            print("deal is not installed. Install it with 'pip install deal'.")
            exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
