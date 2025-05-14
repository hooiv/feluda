"""
Tracing Module

This module provides tracing capabilities using OpenTelemetry.
"""

import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio, TraceIdRatioBased
from opentelemetry.trace import Span, Status, StatusCode

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def setup_tracing(
    service_name: str,
    service_version: str,
    environment: str = "development",
    sampling_ratio: float = 1.0,
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
) -> None:
    """
    Set up OpenTelemetry tracing.
    
    Args:
        service_name: The name of the service.
        service_version: The version of the service.
        environment: The environment (e.g., development, production).
        sampling_ratio: The sampling ratio (0.0 to 1.0).
        otlp_endpoint: The OTLP endpoint for exporting traces.
                      If None, no OTLP exporter is configured.
        console_export: Whether to export traces to the console.
    """
    # Create a resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": environment,
    })
    
    # Create a trace provider with the resource
    trace_provider = TracerProvider(
        resource=resource,
        sampler=ParentBasedTraceIdRatio(TraceIdRatioBased(sampling_ratio)),
    )
    
    # Add exporters
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    if console_export or os.environ.get("FELUDA_TRACE_CONSOLE", "").lower() in ("true", "1", "yes"):
        console_exporter = ConsoleSpanExporter()
        trace_provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Set the trace provider as the global provider
    trace.set_tracer_provider(trace_provider)


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to trace a function.
    
    Args:
        name: The name of the span. If None, the function name is used.
        attributes: Additional attributes to add to the span.
        
    Returns:
        The decorated function.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        span_name = name or func.__name__
        span_attributes = attributes or {}
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            tracer = trace.get_tracer(func.__module__)
            
            with tracer.start_as_current_span(span_name, attributes=span_attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    return decorator


class TracingContextManager:
    """
    Context manager for creating spans.
    
    This class provides a context manager for creating spans and setting them as the current span.
    """
    
    def __init__(
        self,
        name: str,
        tracer_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a TracingContextManager.
        
        Args:
            name: The name of the span.
            tracer_name: The name of the tracer. If None, the caller's module name is used.
            attributes: Additional attributes to add to the span.
        """
        self.name = name
        self.tracer_name = tracer_name or _get_caller_module_name()
        self.attributes = attributes or {}
        self.span: Optional[Span] = None
    
    def __enter__(self) -> Span:
        """
        Enter the context manager.
        
        Returns:
            The created span.
        """
        tracer = trace.get_tracer(self.tracer_name)
        self.span = tracer.start_as_current_span(self.name, attributes=self.attributes)
        return self.span.__enter__()
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context manager.
        
        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The exception traceback, if an exception was raised.
        """
        if self.span:
            if exc_val:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.__exit__(exc_type, exc_val, exc_tb)


def create_span(
    name: str,
    tracer_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> TracingContextManager:
    """
    Create a span context manager.
    
    Args:
        name: The name of the span.
        tracer_name: The name of the tracer. If None, the caller's module name is used.
        attributes: Additional attributes to add to the span.
        
    Returns:
        A context manager for the span.
    """
    return TracingContextManager(name, tracer_name, attributes)


def add_span_attribute(key: str, value: Any) -> None:
    """
    Add an attribute to the current span.
    
    Args:
        key: The attribute key.
        value: The attribute value.
    """
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Add an event to the current span.
    
    Args:
        name: The event name.
        attributes: Additional attributes for the event.
    """
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.add_event(name, attributes)


def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Record an exception in the current span.
    
    Args:
        exception: The exception to record.
        attributes: Additional attributes for the exception.
    """
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.record_exception(exception, attributes)
        current_span.set_status(Status(StatusCode.ERROR, str(exception)))


def _get_caller_module_name() -> str:
    """
    Get the name of the caller's module.
    
    Returns:
        The name of the caller's module.
    """
    import inspect
    frame = inspect.currentframe()
    if frame:
        try:
            frame = frame.f_back
            if frame:
                frame = frame.f_back
                if frame:
                    module = inspect.getmodule(frame)
                    if module:
                        return module.__name__
        finally:
            del frame
    
    return "unknown"
