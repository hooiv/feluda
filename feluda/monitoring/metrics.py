"""
Metrics module for Feluda.

This module provides metrics collection and reporting for Feluda.
"""

import abc
import enum
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import prometheus_client
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class MetricType(str, enum.Enum):
    """Enum for metric types."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Metric(abc.ABC):
    """
    Base class for metrics.
    
    This class defines the interface for metrics.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize a metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            labels: The metric labels.
        """
        self.name = name
        self.description = description
        self.labels = labels or []
    
    @property
    @abc.abstractmethod
    def type(self) -> MetricType:
        """
        Get the metric type.
        
        Returns:
            The metric type.
        """
        pass
    
    @abc.abstractmethod
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the metric value.
        
        Args:
            labels: The metric labels.
            
        Returns:
            The metric value.
        """
        pass


class Counter(Metric):
    """
    Counter metric.
    
    This class represents a counter metric that can only increase.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize a counter metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            labels: The metric labels.
        """
        super().__init__(name, description, labels)
        self._counter = prometheus_client.Counter(
            name=name,
            documentation=description,
            labelnames=labels or [],
        )
    
    @property
    def type(self) -> MetricType:
        """
        Get the metric type.
        
        Returns:
            The metric type.
        """
        return MetricType.COUNTER
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment the counter.
        
        Args:
            value: The value to increment by.
            labels: The metric labels.
        """
        if labels:
            self._counter.labels(**labels).inc(value)
        else:
            self._counter.inc(value)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the counter value.
        
        Args:
            labels: The metric labels.
            
        Returns:
            The counter value.
        """
        if labels:
            return self._counter.labels(**labels)._value.get()
        else:
            return self._counter._value.get()


class Gauge(Metric):
    """
    Gauge metric.
    
    This class represents a gauge metric that can increase and decrease.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize a gauge metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            labels: The metric labels.
        """
        super().__init__(name, description, labels)
        self._gauge = prometheus_client.Gauge(
            name=name,
            documentation=description,
            labelnames=labels or [],
        )
    
    @property
    def type(self) -> MetricType:
        """
        Get the metric type.
        
        Returns:
            The metric type.
        """
        return MetricType.GAUGE
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment the gauge.
        
        Args:
            value: The value to increment by.
            labels: The metric labels.
        """
        if labels:
            self._gauge.labels(**labels).inc(value)
        else:
            self._gauge.inc(value)
    
    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Decrement the gauge.
        
        Args:
            value: The value to decrement by.
            labels: The metric labels.
        """
        if labels:
            self._gauge.labels(**labels).dec(value)
        else:
            self._gauge.dec(value)
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set the gauge value.
        
        Args:
            value: The value to set.
            labels: The metric labels.
        """
        if labels:
            self._gauge.labels(**labels).set(value)
        else:
            self._gauge.set(value)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the gauge value.
        
        Args:
            labels: The metric labels.
            
        Returns:
            The gauge value.
        """
        if labels:
            return self._gauge.labels(**labels)._value.get()
        else:
            return self._gauge._value.get()


class Histogram(Metric):
    """
    Histogram metric.
    
    This class represents a histogram metric that samples observations and counts them in configurable buckets.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize a histogram metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            buckets: The histogram buckets.
            labels: The metric labels.
        """
        super().__init__(name, description, labels)
        self._histogram = prometheus_client.Histogram(
            name=name,
            documentation=description,
            buckets=buckets or prometheus_client.Histogram.DEFAULT_BUCKETS,
            labelnames=labels or [],
        )
    
    @property
    def type(self) -> MetricType:
        """
        Get the metric type.
        
        Returns:
            The metric type.
        """
        return MetricType.HISTOGRAM
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value.
        
        Args:
            value: The value to observe.
            labels: The metric labels.
        """
        if labels:
            self._histogram.labels(**labels).observe(value)
        else:
            self._histogram.observe(value)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the histogram value.
        
        Args:
            labels: The metric labels.
            
        Returns:
            The histogram value.
        """
        if labels:
            return self._histogram.labels(**labels)._sum.get()
        else:
            return self._histogram._sum.get()


class Summary(Metric):
    """
    Summary metric.
    
    This class represents a summary metric that samples observations and calculates quantiles over a sliding time window.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize a summary metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            quantiles: The summary quantiles.
            labels: The metric labels.
        """
        super().__init__(name, description, labels)
        self._summary = prometheus_client.Summary(
            name=name,
            documentation=description,
            labelnames=labels or [],
        )
    
    @property
    def type(self) -> MetricType:
        """
        Get the metric type.
        
        Returns:
            The metric type.
        """
        return MetricType.SUMMARY
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value.
        
        Args:
            value: The value to observe.
            labels: The metric labels.
        """
        if labels:
            self._summary.labels(**labels).observe(value)
        else:
            self._summary.observe(value)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the summary value.
        
        Args:
            labels: The metric labels.
            
        Returns:
            The summary value.
        """
        if labels:
            return self._summary.labels(**labels)._sum.get()
        else:
            return self._summary._sum.get()


class MetricManager:
    """
    Metric manager.
    
    This class is responsible for managing metrics.
    """
    
    def __init__(self):
        """
        Initialize the metric manager.
        """
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.RLock()
        
        # Start the Prometheus HTTP server if metrics are enabled
        config = get_config()
        
        if config.metrics_enabled:
            try:
                prometheus_client.start_http_server(
                    port=int(config.metrics_url.split(":")[-1]) if config.metrics_url else 9090,
                )
                log.info("Started Prometheus HTTP server")
            except Exception as e:
                log.error(f"Failed to start Prometheus HTTP server: {e}")
    
    def register_metric(self, metric: Metric) -> None:
        """
        Register a metric.
        
        Args:
            metric: The metric to register.
        """
        with self.lock:
            self.metrics[metric.name] = metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get a metric by name.
        
        Args:
            name: The metric name.
            
        Returns:
            The metric, or None if the metric is not found.
        """
        with self.lock:
            return self.metrics.get(name)
    
    def get_metrics(self) -> Dict[str, Metric]:
        """
        Get all metrics.
        
        Returns:
            A dictionary mapping metric names to metrics.
        """
        with self.lock:
            return self.metrics.copy()
    
    def create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """
        Create a counter metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            labels: The metric labels.
            
        Returns:
            The counter metric.
        """
        with self.lock:
            metric = self.get_metric(name)
            
            if metric:
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {name} already exists with a different type")
                
                return metric
            
            counter = Counter(name, description, labels)
            self.register_metric(counter)
            
            return counter
    
    def create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """
        Create a gauge metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            labels: The metric labels.
            
        Returns:
            The gauge metric.
        """
        with self.lock:
            metric = self.get_metric(name)
            
            if metric:
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {name} already exists with a different type")
                
                return metric
            
            gauge = Gauge(name, description, labels)
            self.register_metric(gauge)
            
            return gauge
    
    def create_histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ) -> Histogram:
        """
        Create a histogram metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            buckets: The histogram buckets.
            labels: The metric labels.
            
        Returns:
            The histogram metric.
        """
        with self.lock:
            metric = self.get_metric(name)
            
            if metric:
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {name} already exists with a different type")
                
                return metric
            
            histogram = Histogram(name, description, buckets, labels)
            self.register_metric(histogram)
            
            return histogram
    
    def create_summary(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ) -> Summary:
        """
        Create a summary metric.
        
        Args:
            name: The metric name.
            description: The metric description.
            quantiles: The summary quantiles.
            labels: The metric labels.
            
        Returns:
            The summary metric.
        """
        with self.lock:
            metric = self.get_metric(name)
            
            if metric:
                if not isinstance(metric, Summary):
                    raise ValueError(f"Metric {name} already exists with a different type")
                
                return metric
            
            summary = Summary(name, description, quantiles, labels)
            self.register_metric(summary)
            
            return summary


# Global metric manager instance
_metric_manager = None
_metric_manager_lock = threading.RLock()


def get_metric_manager() -> MetricManager:
    """
    Get the global metric manager instance.
    
    Returns:
        The global metric manager instance.
    """
    global _metric_manager
    
    with _metric_manager_lock:
        if _metric_manager is None:
            _metric_manager = MetricManager()
        
        return _metric_manager
