"""
Feluda: A configurable engine for analyzing multi-lingual and multi-modal content.

This package provides tools for analyzing data collected from social media - images, text, and video.
It forms the core of search, clustering, and analysis services.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("feluda")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.9.4"  # Default version if package is not installed

from .feluda import Feluda

# Import subpackages
from feluda import ai_agents
from feluda import autonomic
from feluda import crypto
from feluda import exceptions
from feluda import hardware
from feluda import models
from feluda import observability
from feluda import performance
from feluda import resilience
from feluda import testing
from feluda import verification

# Import base operator
from feluda.base_operator import BaseFeludaOperator

__all__ = [
    # Core
    "Feluda",
    "BaseFeludaOperator",
    "__version__",

    # Subpackages
    "ai_agents",
    "autonomic",
    "crypto",
    "exceptions",
    "hardware",
    "models",
    "observability",
    "performance",
    "resilience",
    "testing",
    "verification",
]
