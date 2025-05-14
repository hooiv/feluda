"""
Structured Logging Module

This module provides structured logging capabilities using structlog and OpenTelemetry.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

import structlog
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class FeludaLogger:
    """
    Structured logger for Feluda.
    
    This class provides structured logging capabilities using structlog and OpenTelemetry.
    It automatically adds context information to log messages, such as trace IDs and span IDs.
    """
    
    def __init__(self, name: str):
        """
        Initialize a FeludaLogger.
        
        Args:
            name: The name of the logger.
        """
        self.name = name
        self.logger = structlog.get_logger(name)
        self.tracer = trace.get_tracer(name)
    
    def bind(self, **kwargs: Any) -> "FeludaLogger":
        """
        Bind additional context to the logger.
        
        Args:
            **kwargs: Key-value pairs to add to the context.
            
        Returns:
            A new logger with the additional context.
        """
        new_logger = FeludaLogger(self.name)
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger
    
    def _add_span_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add span context to the log record.
        
        Args:
            kwargs: The keyword arguments to add to.
            
        Returns:
            The updated keyword arguments.
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            span_context = current_span.get_span_context()
            kwargs["trace_id"] = format(span_context.trace_id, "032x")
            kwargs["span_id"] = format(span_context.span_id, "016x")
        return kwargs
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log a debug message.
        
        Args:
            message: The message to log.
            **kwargs: Additional context to add to the log record.
        """
        kwargs = self._add_span_context(kwargs)
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log an info message.
        
        Args:
            message: The message to log.
            **kwargs: Additional context to add to the log record.
        """
        kwargs = self._add_span_context(kwargs)
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log a warning message.
        
        Args:
            message: The message to log.
            **kwargs: Additional context to add to the log record.
        """
        kwargs = self._add_span_context(kwargs)
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log an error message.
        
        Args:
            message: The message to log.
            **kwargs: Additional context to add to the log record.
        """
        kwargs = self._add_span_context(kwargs)
        self.logger.error(message, **kwargs)
    
    def exception(self, message: str, exc_info: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log an exception.
        
        Args:
            message: The message to log.
            exc_info: The exception info. If None, sys.exc_info() is used.
            **kwargs: Additional context to add to the log record.
        """
        kwargs = self._add_span_context(kwargs)
        self.logger.exception(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log a critical message.
        
        Args:
            message: The message to log.
            **kwargs: Additional context to add to the log record.
        """
        kwargs = self._add_span_context(kwargs)
        self.logger.critical(message, **kwargs)
    
    def trace_method(self, name: Optional[str] = None, **attributes: Any):
        """
        Decorator to trace a method.
        
        This decorator creates a span for the method and logs the start and end of the method.
        
        Args:
            name: The name of the span. If None, the method name is used.
            **attributes: Additional attributes to add to the span.
            
        Returns:
            The decorated method.
        """
        def decorator(func):
            span_name = name or func.__name__
            
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(span_name, attributes=attributes) as span:
                    start_time = time.time()
                    self.debug(f"Starting {span_name}", **attributes)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        self.exception(f"Error in {span_name}", exc_info=e, **attributes)
                        raise
                    finally:
                        end_time = time.time()
                        duration_ms = (end_time - start_time) * 1000
                        self.debug(
                            f"Finished {span_name}",
                            duration_ms=duration_ms,
                            **attributes
                        )
            
            return wrapper
        
        return decorator


def get_logger(name: str) -> FeludaLogger:
    """
    Get a logger with the given name.
    
    Args:
        name: The name of the logger.
        
    Returns:
        A FeludaLogger instance.
    """
    return FeludaLogger(name)


# Configure the root logger
logging.basicConfig(
    level=os.environ.get("FELUDA_LOG_LEVEL", "INFO"),
    format="%(message)s",
    stream=sys.stdout,
)


# Set the log level for third-party libraries
for logger_name in ["urllib3", "requests", "boto3", "botocore", "s3transfer"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
