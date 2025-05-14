#!/usr/bin/env python
"""
Setup script for Feluda.
"""

import os
from setuptools import setup, find_packages

# Read the version from the package
with open(os.path.join("feluda", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.0.0"

# Read the long description from the README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define the dependencies
core_dependencies = [
    "pydantic>=2.0.0",
    "deal>=4.24.1",
    "numpy>=1.20.0",
    "structlog>=23.1.0",
    "opentelemetry-api>=1.18.0",
    "opentelemetry-sdk>=1.18.0",
    "opentelemetry-exporter-otlp>=1.18.0",
    "prometheus-client>=0.17.0",
    "boto3>=1.37.0",
    "dacite>=1.9.0",
    "pydub>=0.25.0",
    "pyyaml>=6.0.0",
    "requests>=2.32.0",
    "werkzeug>=3.1.0",
    "wget>=3.2",
    "pillow>=11.0.0",
]

performance_dependencies = [
    "numba>=0.57.0",
]

crypto_dependencies = [
    "pyfhel>=3.0.0",
    "phe>=1.5.0",
]

ai_agents_dependencies = [
    "openai>=1.0.0",
    "langchain>=0.0.300",
    "langchain-openai>=0.0.1",
]

testing_dependencies = [
    "crosshair-tool>=0.0.37",
    "hypothesis>=6.82.0",
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
]

autonomic_dependencies = [
    "scikit-optimize>=0.9.0",
    "scikit-learn>=1.3.0",
]

dev_dependencies = [
    "black>=23.7.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pylint>=2.17.5",
    "pre-commit>=3.3.3",
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "sphinx-autodoc-typehints>=1.24.0",
    "pip-tools>=7.3.0",
    "tomlkit>=0.12.0",
    "ruff>=0.1.0",
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "mutmut>=2.4.0",
    "mypyc>=1.5.0",
]

security_dependencies = [
    "bandit>=1.7.5",
    "safety>=2.3.0",
    "pip-audit>=2.6.0",
]

ml_dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]

hardware_dependencies = [
    "qiskit>=0.44.0",
    "cirq>=1.2.0",
    "pennylane>=0.30.0",
]

neuromorphic_dependencies = [
    "nengo>=3.2.0",
]

# Define the extras
extras_require = {
    "performance": performance_dependencies,
    "crypto": crypto_dependencies,
    "ai_agents": ai_agents_dependencies,
    "testing": testing_dependencies,
    "autonomic": autonomic_dependencies,
    "dev": dev_dependencies + security_dependencies,
    "security": security_dependencies,
    "ml": ml_dependencies,
    "hardware": hardware_dependencies,
    "neuromorphic": neuromorphic_dependencies,
    "all": (
        performance_dependencies
        + crypto_dependencies
        + ai_agents_dependencies
        + testing_dependencies
        + autonomic_dependencies
        + ml_dependencies
    ),
    "full": (
        performance_dependencies
        + crypto_dependencies
        + ai_agents_dependencies
        + testing_dependencies
        + autonomic_dependencies
        + dev_dependencies
        + security_dependencies
        + ml_dependencies
        + hardware_dependencies
        + neuromorphic_dependencies
    ),
}

setup(
    name="feluda",
    version=version,
    description="A configurable engine for analysing multi-lingual and multi-modal content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tattle",
    author_email="tech@tattle.co.in",
    url="https://github.com/tattle-made/feluda",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=core_dependencies,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, computer vision, multimedia, analysis, search, ai",
    project_urls={
        "Bug Reports": "https://github.com/tattle-made/feluda/issues",
        "Source": "https://github.com/tattle-made/feluda",
        "Documentation": "https://github.com/tattle-made/feluda/wiki",
    },
)
