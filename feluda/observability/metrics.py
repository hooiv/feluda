"""
Metrics Module

This module provides metrics collection capabilities using OpenTelemetry.
"""

import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def setup_metrics(
    service_name: str,
    service_version: str,
    environment: str = "development",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
    export_interval_ms: int = 30000,
) -> None:
    """
    Set up OpenTelemetry metrics.
    
    Args:
        service_name: The name of the service.
        service_version: The version of the service.
        environment: The environment (e.g., development, production).
        otlp_endpoint: The OTLP endpoint for exporting metrics.
                      If None, no OTLP exporter is configured.
        console_export: Whether to export metrics to the console.
        export_interval_ms: The interval in milliseconds for exporting metrics.
    """
    # Create a resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": environment,
    })
    
    # Create metric readers
    metric_readers = []
    
    if otlp_endpoint:
        otlp_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
        otlp_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=export_interval_ms,
        )
        metric_readers.append(otlp_reader)
    
    if console_export or os.environ.get("FELUDA_METRICS_CONSOLE", "").lower() in ("true", "1", "yes"):
        console_exporter = ConsoleMetricExporter()
        console_reader = PeriodicExportingMetricReader(
            exporter=console_exporter,
            export_interval_millis=export_interval_ms,
        )
        metric_readers.append(console_reader)
    
    # Create a meter provider with the resource and readers
    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    
    # Set the meter provider as the global provider
    metrics.set_meter_provider(meter_provider)


class MetricsRegistry:
    """
    Registry for metrics.
    
    This class provides a registry for metrics, ensuring that metrics with the same name
    are only created once.
    """
    
    _counters: Dict[str, metrics.Counter] = {}
    _histograms: Dict[str, metrics.Histogram] = {}
    _gauges: Dict[str, metrics.ObservableGauge] = {}
    _up_down_counters: Dict[str, metrics.UpDownCounter] = {}
    
    @classmethod
    def get_counter(
        cls,
        name: str,
        description: str,
        unit: str = "1",
        meter_name: Optional[str] = None,
    ) -> metrics.Counter:
        """
        Get or create a counter.
        
        Args:
            name: The name of the counter.
            description: The description of the counter.
            unit: The unit of the counter.
            meter_name: The name of the meter. If None, the name is used.
            
        Returns:
            The counter.
        """
        if name not in cls._counters:
            meter = metrics.get_meter(meter_name or name)
            cls._counters[name] = meter.create_counter(name, description, unit)
        
        return cls._counters[name]
    
    @classmethod
    def get_histogram(
        cls,
        name: str,
        description: str,
        unit: str = "1",
        meter_name: Optional[str] = None,
    ) -> metrics.Histogram:
        """
        Get or create a histogram.
        
        Args:
            name: The name of the histogram.
            description: The description of the histogram.
            unit: The unit of the histogram.
            meter_name: The name of the meter. If None, the name is used.
            
        Returns:
            The histogram.
        """
        if name not in cls._histograms:
            meter = metrics.get_meter(meter_name or name)
            cls._histograms[name] = meter.create_histogram(name, description, unit)
        
        return cls._histograms[name]
    
    @classmethod
    def get_up_down_counter(
        cls,
        name: str,
        description: str,
        unit: str = "1",
        meter_name: Optional[str] = None,
    ) -> metrics.UpDownCounter:
        """
        Get or create an up-down counter.
        
        Args:
            name: The name of the up-down counter.
            description: The description of the up-down counter.
            unit: The unit of the up-down counter.
            meter_name: The name of the meter. If None, the name is used.
            
        Returns:
            The up-down counter.
        """
        if name not in cls._up_down_counters:
            meter = metrics.get_meter(meter_name or name)
            cls._up_down_counters[name] = meter.create_up_down_counter(name, description, unit)
        
        return cls._up_down_counters[name]


def count_calls(
    name: str,
    description: str,
    unit: str = "1",
    meter_name: Optional[str] = None,
    attributes_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to count calls to a function.
    
    Args:
        name: The name of the counter.
        description: The description of the counter.
        unit: The unit of the counter.
        meter_name: The name of the meter. If None, the name is used.
        attributes_fn: A function that returns attributes for the counter.
                      The function is called with the same arguments as the decorated function.
        
    Returns:
        The decorated function.
    """
    counter = MetricsRegistry.get_counter(name, description, unit, meter_name)
    
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            attributes = {}
            if attributes_fn:
                attributes = attributes_fn(*args, **kwargs)
            
            counter.add(1, attributes)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def measure_execution_time(
    name: str,
    description: str,
    unit: str = "ms",
    meter_name: Optional[str] = None,
    attributes_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to measure the execution time of a function.
    
    Args:
        name: The name of the histogram.
        description: The description of the histogram.
        unit: The unit of the histogram.
        meter_name: The name of the meter. If None, the name is used.
        attributes_fn: A function that returns attributes for the histogram.
                      The function is called with the same arguments as the decorated function.
        
    Returns:
        The decorated function.
    """
    histogram = MetricsRegistry.get_histogram(name, description, unit, meter_name)
    
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            attributes = {}
            if attributes_fn:
                attributes = attributes_fn(*args, **kwargs)
            
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                histogram.record(duration_ms, attributes)
        
        return wrapper
    
    return decorator


def track_in_progress(
    name: str,
    description: str,
    unit: str = "1",
    meter_name: Optional[str] = None,
    attributes_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to track the number of in-progress calls to a function.
    
    Args:
        name: The name of the up-down counter.
        description: The description of the up-down counter.
        unit: The unit of the up-down counter.
        meter_name: The name of the meter. If None, the name is used.
        attributes_fn: A function that returns attributes for the up-down counter.
                      The function is called with the same arguments as the decorated function.
        
    Returns:
        The decorated function.
    """
    up_down_counter = MetricsRegistry.get_up_down_counter(name, description, unit, meter_name)
    
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            attributes = {}
            if attributes_fn:
                attributes = attributes_fn(*args, **kwargs)
            
            up_down_counter.add(1, attributes)
            try:
                return func(*args, **kwargs)
            finally:
                up_down_counter.add(-1, attributes)
        
        return wrapper
    
    return decorator
