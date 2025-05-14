"""
Observability Package

This package provides observability capabilities for Feluda, including logging, tracing, and metrics.
"""

from feluda.observability.logging import FeludaLogger, get_logger
from feluda.observability.metrics import (
    MetricsRegistry,
    count_calls,
    measure_execution_time,
    setup_metrics,
    track_in_progress,
)
from feluda.observability.tracing import (
    add_span_attribute,
    add_span_event,
    create_span,
    record_exception,
    setup_tracing,
    trace_function,
)

__all__ = [
    # Logging
    "FeludaLogger",
    "get_logger",
    
    # Tracing
    "setup_tracing",
    "trace_function",
    "create_span",
    "add_span_attribute",
    "add_span_event",
    "record_exception",
    
    # Metrics
    "setup_metrics",
    "MetricsRegistry",
    "count_calls",
    "measure_execution_time",
    "track_in_progress",
]
