"""
Tests for the observability package.
"""

import unittest
from unittest.mock import MagicMock, patch

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

from feluda.observability import (
    FeludaLogger,
    MetricsRegistry,
    add_span_attribute,
    add_span_event,
    count_calls,
    create_span,
    get_logger,
    measure_execution_time,
    record_exception,
    setup_metrics,
    setup_tracing,
    trace_function,
    track_in_progress,
)


class TestLogging(unittest.TestCase):
    """Tests for the logging module."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_logger")
        self.assertIsInstance(logger, FeludaLogger)
        self.assertEqual(logger.name, "test_logger")
    
    def test_bind(self):
        """Test bind method."""
        logger = get_logger("test_logger")
        bound_logger = logger.bind(key="value")
        self.assertIsInstance(bound_logger, FeludaLogger)
        self.assertEqual(bound_logger.name, "test_logger")
    
    @patch("feluda.observability.logging.trace.get_current_span")
    def test_add_span_context(self, mock_get_current_span):
        """Test _add_span_context method."""
        # Mock the current span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span_context = MagicMock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context
        mock_get_current_span.return_value = mock_span
        
        # Test adding span context
        logger = get_logger("test_logger")
        kwargs = {}
        result = logger._add_span_context(kwargs)
        
        # Check that the span context was added
        self.assertIn("trace_id", result)
        self.assertIn("span_id", result)
        self.assertEqual(result["trace_id"], "00000000000000000000000075bcd15")
        self.assertEqual(result["span_id"], "000000000000eafd")
    
    @patch("feluda.observability.logging.FeludaLogger._add_span_context")
    @patch("structlog.get_logger")
    def test_log_methods(self, mock_get_logger, mock_add_span_context):
        """Test log methods."""
        # Mock the structlog logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock _add_span_context
        mock_add_span_context.side_effect = lambda kwargs: kwargs
        
        # Test log methods
        logger = get_logger("test_logger")
        
        logger.debug("debug message", key="value")
        mock_logger.debug.assert_called_once_with("debug message", key="value")
        
        logger.info("info message", key="value")
        mock_logger.info.assert_called_once_with("info message", key="value")
        
        logger.warning("warning message", key="value")
        mock_logger.warning.assert_called_once_with("warning message", key="value")
        
        logger.error("error message", key="value")
        mock_logger.error.assert_called_once_with("error message", key="value")
        
        logger.critical("critical message", key="value")
        mock_logger.critical.assert_called_once_with("critical message", key="value")
        
        exception = Exception("test exception")
        logger.exception("exception message", exc_info=exception, key="value")
        mock_logger.exception.assert_called_once_with("exception message", exc_info=exception, key="value")


class TestTracing(unittest.TestCase):
    """Tests for the tracing module."""
    
    @patch("feluda.observability.tracing.trace.set_tracer_provider")
    @patch("feluda.observability.tracing.TracerProvider")
    @patch("feluda.observability.tracing.Resource.create")
    def test_setup_tracing(self, mock_resource_create, mock_tracer_provider, mock_set_tracer_provider):
        """Test setup_tracing function."""
        # Mock the resource
        mock_resource = MagicMock()
        mock_resource_create.return_value = mock_resource
        
        # Mock the tracer provider
        mock_provider = MagicMock()
        mock_tracer_provider.return_value = mock_provider
        
        # Test setup_tracing
        setup_tracing("test_service", "1.0.0")
        
        # Check that the resource was created with the correct attributes
        mock_resource_create.assert_called_once()
        resource_args = mock_resource_create.call_args[0][0]
        self.assertEqual(resource_args["service.name"], "test_service")
        self.assertEqual(resource_args["service.version"], "1.0.0")
        self.assertEqual(resource_args["deployment.environment"], "development")
        
        # Check that the tracer provider was created with the resource
        mock_tracer_provider.assert_called_once()
        
        # Check that the tracer provider was set as the global provider
        mock_set_tracer_provider.assert_called_once_with(mock_provider)
    
    @patch("feluda.observability.tracing.trace.get_tracer")
    def test_trace_function(self, mock_get_tracer):
        """Test trace_function decorator."""
        # Mock the tracer
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        # Mock the span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        
        # Test trace_function decorator
        @trace_function("test_span")
        def test_func(x, y):
            return x + y
        
        # Call the decorated function
        result = test_func(1, 2)
        
        # Check that the tracer was created with the correct module name
        mock_get_tracer.assert_called_once()
        
        # Check that the span was created with the correct name and attributes
        mock_tracer.start_as_current_span.assert_called_once_with("test_span", attributes={})
        
        # Check that the span was used as a context manager
        mock_span.__enter__.assert_called_once()
        mock_span.__exit__.assert_called_once()
        
        # Check that the function was called and returned the correct result
        self.assertEqual(result, 3)
    
    @patch("feluda.observability.tracing.trace.get_tracer")
    def test_create_span(self, mock_get_tracer):
        """Test create_span function."""
        # Mock the tracer
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        # Mock the span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        
        # Test create_span function
        with create_span("test_span", attributes={"key": "value"}) as span:
            # Check that the span was created with the correct name and attributes
            mock_tracer.start_as_current_span.assert_called_once_with("test_span", attributes={"key": "value"})
            
            # Check that the span was used as a context manager
            mock_span.__enter__.assert_called_once()
            
            # Check that the span was returned
            self.assertEqual(span, mock_span.__enter__.return_value)
        
        # Check that the span context manager was exited
        mock_span.__exit__.assert_called_once()
    
    @patch("feluda.observability.tracing.trace.get_current_span")
    def test_add_span_attribute(self, mock_get_current_span):
        """Test add_span_attribute function."""
        # Mock the current span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span
        
        # Test add_span_attribute function
        add_span_attribute("key", "value")
        
        # Check that the attribute was added to the span
        mock_span.set_attribute.assert_called_once_with("key", "value")
    
    @patch("feluda.observability.tracing.trace.get_current_span")
    def test_add_span_event(self, mock_get_current_span):
        """Test add_span_event function."""
        # Mock the current span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span
        
        # Test add_span_event function
        add_span_event("test_event", {"key": "value"})
        
        # Check that the event was added to the span
        mock_span.add_event.assert_called_once_with("test_event", {"key": "value"})
    
    @patch("feluda.observability.tracing.trace.get_current_span")
    def test_record_exception(self, mock_get_current_span):
        """Test record_exception function."""
        # Mock the current span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span
        
        # Test record_exception function
        exception = Exception("test exception")
        record_exception(exception, {"key": "value"})
        
        # Check that the exception was recorded in the span
        mock_span.record_exception.assert_called_once_with(exception, {"key": "value"})
        mock_span.set_status.assert_called_once()


class TestMetrics(unittest.TestCase):
    """Tests for the metrics module."""
    
    @patch("feluda.observability.metrics.metrics.set_meter_provider")
    @patch("feluda.observability.metrics.MeterProvider")
    @patch("feluda.observability.metrics.Resource.create")
    def test_setup_metrics(self, mock_resource_create, mock_meter_provider, mock_set_meter_provider):
        """Test setup_metrics function."""
        # Mock the resource
        mock_resource = MagicMock()
        mock_resource_create.return_value = mock_resource
        
        # Mock the meter provider
        mock_provider = MagicMock()
        mock_meter_provider.return_value = mock_provider
        
        # Test setup_metrics
        setup_metrics("test_service", "1.0.0")
        
        # Check that the resource was created with the correct attributes
        mock_resource_create.assert_called_once()
        resource_args = mock_resource_create.call_args[0][0]
        self.assertEqual(resource_args["service.name"], "test_service")
        self.assertEqual(resource_args["service.version"], "1.0.0")
        self.assertEqual(resource_args["deployment.environment"], "development")
        
        # Check that the meter provider was created with the resource
        mock_meter_provider.assert_called_once()
        
        # Check that the meter provider was set as the global provider
        mock_set_meter_provider.assert_called_once_with(mock_provider)
    
    @patch("feluda.observability.metrics.metrics.get_meter")
    def test_metrics_registry(self, mock_get_meter):
        """Test MetricsRegistry class."""
        # Mock the meter
        mock_meter = MagicMock()
        mock_get_meter.return_value = mock_meter
        
        # Mock the counter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        
        # Test get_counter
        counter1 = MetricsRegistry.get_counter("test_counter", "Test counter")
        
        # Check that the meter was created with the correct name
        mock_get_meter.assert_called_once_with("test_counter")
        
        # Check that the counter was created with the correct name and description
        mock_meter.create_counter.assert_called_once_with("test_counter", "Test counter", "1")
        
        # Check that the counter was returned
        self.assertEqual(counter1, mock_counter)
        
        # Test get_counter again with the same name
        counter2 = MetricsRegistry.get_counter("test_counter", "Test counter")
        
        # Check that the meter and counter were not created again
        self.assertEqual(mock_get_meter.call_count, 1)
        self.assertEqual(mock_meter.create_counter.call_count, 1)
        
        # Check that the same counter was returned
        self.assertEqual(counter2, mock_counter)
    
    @patch("feluda.observability.metrics.MetricsRegistry.get_counter")
    def test_count_calls(self, mock_get_counter):
        """Test count_calls decorator."""
        # Mock the counter
        mock_counter = MagicMock()
        mock_get_counter.return_value = mock_counter
        
        # Test count_calls decorator
        @count_calls("test_counter", "Test counter")
        def test_func(x, y):
            return x + y
        
        # Call the decorated function
        result = test_func(1, 2)
        
        # Check that the counter was created with the correct name and description
        mock_get_counter.assert_called_once_with("test_counter", "Test counter", "1", None)
        
        # Check that the counter was incremented
        mock_counter.add.assert_called_once_with(1, {})
        
        # Check that the function was called and returned the correct result
        self.assertEqual(result, 3)
    
    @patch("feluda.observability.metrics.MetricsRegistry.get_histogram")
    def test_measure_execution_time(self, mock_get_histogram):
        """Test measure_execution_time decorator."""
        # Mock the histogram
        mock_histogram = MagicMock()
        mock_get_histogram.return_value = mock_histogram
        
        # Test measure_execution_time decorator
        @measure_execution_time("test_histogram", "Test histogram")
        def test_func(x, y):
            return x + y
        
        # Call the decorated function
        result = test_func(1, 2)
        
        # Check that the histogram was created with the correct name and description
        mock_get_histogram.assert_called_once_with("test_histogram", "Test histogram", "ms", None)
        
        # Check that the histogram recorded the execution time
        mock_histogram.record.assert_called_once()
        
        # Check that the function was called and returned the correct result
        self.assertEqual(result, 3)
    
    @patch("feluda.observability.metrics.MetricsRegistry.get_up_down_counter")
    def test_track_in_progress(self, mock_get_up_down_counter):
        """Test track_in_progress decorator."""
        # Mock the up-down counter
        mock_up_down_counter = MagicMock()
        mock_get_up_down_counter.return_value = mock_up_down_counter
        
        # Test track_in_progress decorator
        @track_in_progress("test_counter", "Test counter")
        def test_func(x, y):
            return x + y
        
        # Call the decorated function
        result = test_func(1, 2)
        
        # Check that the up-down counter was created with the correct name and description
        mock_get_up_down_counter.assert_called_once_with("test_counter", "Test counter", "1", None)
        
        # Check that the up-down counter was incremented and decremented
        mock_up_down_counter.add.assert_any_call(1, {})
        mock_up_down_counter.add.assert_any_call(-1, {})
        
        # Check that the function was called and returned the correct result
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
