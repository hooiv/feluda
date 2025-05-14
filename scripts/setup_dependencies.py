#!/usr/bin/env python
"""
Setup Dependencies Script

This script installs the dependencies required for Feluda v1.0.0.
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("setup_dependencies")


# Define dependency groups
DEPENDENCY_GROUPS = {
    "core": [
        "pydantic>=2.0.0",
        "deal>=4.24.1",
        "numpy>=1.20.0",
        "structlog>=23.1.0",
    ],
    "observability": [
        "opentelemetry-api>=1.18.0",
        "opentelemetry-sdk>=1.18.0",
        "opentelemetry-exporter-otlp>=1.18.0",
        "prometheus-client>=0.17.0",
    ],
    "performance": [
        "numba>=0.57.0",
        "torch>=2.0.0",
    ],
    "crypto": [
        "pyfhel>=3.0.0",
        "tenseal>=0.3.12",
        "phe>=1.5.0",
    ],
    "ai_agents": [
        "openai>=1.0.0",
        "langchain>=0.0.300",
        "langchain-openai>=0.0.1",
    ],
    "testing": [
        "crosshair-tool>=0.0.37",
        "hypothesis>=6.82.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    "hardware": [
        "qiskit>=0.44.0",
        "cirq>=1.2.0",
        "pennylane>=0.30.0",
        "nengo>=3.2.0",
    ],
    "autonomic": [
        "scikit-optimize>=0.9.0",
        "scikit-learn>=1.3.0",
    ],
    "dev": [
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "pylint>=2.17.5",
        "pre-commit>=3.3.3",
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
}


def check_python_version() -> bool:
    """
    Check if the Python version is compatible.
    
    Returns:
        True if the Python version is compatible, False otherwise.
    """
    required_version = (3, 10)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        log.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        log.error(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    return True


def check_pip() -> bool:
    """
    Check if pip is available.
    
    Returns:
        True if pip is available, False otherwise.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.SubprocessError:
        log.error("pip is not available")
        return False


def install_dependencies(
    groups: Optional[List[str]] = None,
    upgrade: bool = False,
    use_uv: bool = False,
) -> bool:
    """
    Install dependencies.
    
    Args:
        groups: The dependency groups to install. If None, all groups are installed.
        upgrade: Whether to upgrade existing packages.
        use_uv: Whether to use uv instead of pip.
        
    Returns:
        True if installation was successful, False otherwise.
    """
    if groups is None:
        groups = list(DEPENDENCY_GROUPS.keys())
    
    # Collect dependencies
    dependencies = []
    for group in groups:
        if group in DEPENDENCY_GROUPS:
            dependencies.extend(DEPENDENCY_GROUPS[group])
        else:
            log.warning(f"Unknown dependency group: {group}")
    
    if not dependencies:
        log.warning("No dependencies to install")
        return True
    
    # Remove duplicates
    dependencies = list(set(dependencies))
    
    # Install dependencies
    log.info(f"Installing {len(dependencies)} dependencies")
    
    if use_uv:
        # Use uv
        cmd = ["uv", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(dependencies)
        
        log.info(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(
                cmd,
                check=True,
            )
            return True
        except subprocess.SubprocessError as e:
            log.error(f"Failed to install dependencies with uv: {e}")
            return False
    else:
        # Use pip
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(dependencies)
        
        log.info(f"Running: {' '.join(cmd)}")
        
        try:
            subprocess.run(
                cmd,
                check=True,
            )
            return True
        except subprocess.SubprocessError as e:
            log.error(f"Failed to install dependencies with pip: {e}")
            return False


def install_optional_dependencies(
    upgrade: bool = False,
    use_uv: bool = False,
) -> bool:
    """
    Install optional dependencies.
    
    Args:
        upgrade: Whether to upgrade existing packages.
        use_uv: Whether to use uv instead of pip.
        
    Returns:
        True if installation was successful, False otherwise.
    """
    # Try to install optional dependencies
    optional_dependencies = [
        "tensorflow>=2.13.0",  # For hardware acceleration
        "jax>=0.4.14",         # For hardware acceleration
        "pyquil>=4.0.0",       # For quantum computing
        "braket>=1.36.0",      # For quantum computing
        "qsharp>=0.28.302812", # For quantum computing
        "brian2>=2.5.1",       # For neuromorphic computing
        "nest-simulator>=3.4", # For neuromorphic computing
        "nengo-dl>=3.6.0",     # For neuromorphic computing
        "nengo-loihi>=1.0.0",  # For neuromorphic computing
    ]
    
    log.info("Installing optional dependencies")
    log.info("These may fail to install on some platforms")
    
    if use_uv:
        # Use uv
        for dependency in optional_dependencies:
            cmd = ["uv", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(dependency)
            
            log.info(f"Running: {' '.join(cmd)}")
            
            try:
                subprocess.run(
                    cmd,
                    check=True,
                )
            except subprocess.SubprocessError as e:
                log.warning(f"Failed to install optional dependency {dependency}: {e}")
    else:
        # Use pip
        for dependency in optional_dependencies:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(dependency)
            
            log.info(f"Running: {' '.join(cmd)}")
            
            try:
                subprocess.run(
                    cmd,
                    check=True,
                )
            except subprocess.SubprocessError as e:
                log.warning(f"Failed to install optional dependency {dependency}: {e}")
    
    return True


def setup_development_environment() -> bool:
    """
    Set up the development environment.
    
    Returns:
        True if setup was successful, False otherwise.
    """
    # Install pre-commit hooks
    log.info("Setting up pre-commit hooks")
    
    try:
        subprocess.run(
            ["pre-commit", "install"],
            check=True,
        )
    except subprocess.SubprocessError as e:
        log.error(f"Failed to set up pre-commit hooks: {e}")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up dependencies for Feluda")
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        choices=list(DEPENDENCY_GROUPS.keys()) + ["all"],
        default=["all"],
        help="The dependency groups to install",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade existing packages",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Use uv instead of pip",
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Set up the development environment",
    )
    
    args = parser.parse_args()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check pip
    if not check_pip():
        return 1
    
    # Determine groups to install
    groups = args.groups
    if "all" in groups:
        groups = list(DEPENDENCY_GROUPS.keys())
    
    # Install dependencies
    if not install_dependencies(groups, args.upgrade, args.use_uv):
        return 1
    
    # Install optional dependencies
    if args.optional:
        if not install_optional_dependencies(args.upgrade, args.use_uv):
            return 1
    
    # Set up development environment
    if args.dev:
        if not setup_development_environment():
            return 1
    
    log.info("Dependencies installed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
