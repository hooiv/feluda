#!/usr/bin/env python
"""
Advanced Features Demo

This script demonstrates the advanced features of Feluda v1.0.0.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feluda.autonomic.ml_tuning import (
    OptimizationAlgorithm,
    OptimizationConfig,
    Parameter,
    ParameterType,
    optimize_parameters,
)
from feluda.autonomic.self_healing import (
    HealingAction,
    HealingStrategy,
    HealthCheck,
    HealthStatus,
    SelfHealingSystem,
)
from feluda.base_operator import BaseFeludaOperator
from feluda.hardware.fpga import (
    FPGAConfig,
    FPGAFamily,
    FPGAInterface,
    FPGAVendor,
    HDLLanguage,
)
from feluda.models.data_models import MediaContent, MediaMetadata, MediaType, OperatorResult
from feluda.observability import get_logger, trace_function, add_span_attribute, count_calls, measure_execution_time
from feluda.performance import optional_njit
from feluda.resilience.circuit_breaker import circuit_breaker
from feluda.testing.chaos import chaos_monkey, chaos_context
from feluda.testing.fuzzing import (
    FuzzingConfig,
    FuzzingStrategy,
    JSON_GRAMMAR,
    fuzz_function,
)
from feluda.testing.metamorphic import (
    EqualityRelation,
    run_metamorphic_tests,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = get_logger("advanced_features_demo")


# Define a sample operator using BaseFeludaOperator
class ImageProcessor(BaseFeludaOperator[MediaContent, Dict[str, Any], Dict[str, Any]]):
    """
    Sample operator for processing images.
    
    This operator demonstrates the use of the BaseFeludaOperator class with contracts.
    """
    
    name = "ImageProcessor"
    description = "Process images using various techniques."
    version = "1.0.0"
    parameters_model = Dict[str, Any]  # For simplicity, we use a dict for parameters
    
    def _initialize(self) -> None:
        """Initialize the operator."""
        log.info("Initializing ImageProcessor operator")
        
        # Set default parameters if not provided
        if not self.parameters:
            self.parameters = {
                "resize_width": 224,
                "resize_height": 224,
                "normalize": True,
            }
    
    def _validate_input(self, input_data: MediaContent) -> bool:
        """Validate the input data."""
        if not isinstance(input_data, MediaContent):
            return False
        
        if input_data.metadata.media_type != MediaType.IMAGE:
            return False
        
        return input_data.content_data is not None or input_data.content_uri is not None
    
    @trace_function(name="image_processor_execute")
    def _execute(self, input_data: MediaContent) -> Dict[str, Any]:
        """Execute the operator on the input data."""
        log.info(f"Processing image: {input_data.metadata.media_id}")
        
        # Add span attributes for observability
        add_span_attribute("image_id", input_data.metadata.media_id)
        add_span_attribute("resize_width", self.parameters["resize_width"])
        add_span_attribute("resize_height", self.parameters["resize_height"])
        
        # Simulate image processing
        # In a real implementation, this would use a library like PIL or OpenCV
        result = {
            "image_id": input_data.metadata.media_id,
            "width": self.parameters["resize_width"],
            "height": self.parameters["resize_height"],
            "normalized": self.parameters["normalize"],
            "features": self._extract_features(input_data),
        }
        
        return result
    
    @optional_njit  # Use Numba JIT compilation if available
    def _extract_features(self, input_data: MediaContent) -> List[float]:
        """Extract features from the image."""
        # Simulate feature extraction
        # In a real implementation, this would use a deep learning model
        return [0.1, 0.2, 0.3, 0.4, 0.5]


# Define a function with a circuit breaker
@circuit_breaker(
    name="external_service",
    failure_threshold=3,
    recovery_timeout=10.0,
    expected_exceptions=[ConnectionError, TimeoutError],
)
def call_external_service() -> str:
    """
    Call an external service that might fail.
    
    This function demonstrates the use of the circuit breaker pattern.
    """
    # Simulate a service call that might fail
    if np.random.random() < 0.5:
        raise ConnectionError("Failed to connect to external service")
    
    return "Response from external service"


# Define a function with chaos monkey
@chaos_monkey(failure_probability=0.2)
@count_calls(name="process_data_calls", description="Number of calls to process_data")
@measure_execution_time(name="process_data_duration", description="Duration of process_data", unit="ms")
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process data with potential chaos.
    
    This function demonstrates the use of the chaos monkey pattern.
    """
    # Process the data
    result = {
        "processed": True,
        "input_size": len(data),
        "timestamp": time.time(),
    }
    
    return result


# Define a function for metamorphic testing
def sort_list(lst: List[int]) -> List[int]:
    """
    Sort a list of integers.
    
    This function is used to demonstrate metamorphic testing.
    """
    return sorted(lst)


# Define a function for fuzzing
def parse_json(json_str: str) -> Dict[str, Any]:
    """
    Parse a JSON string.
    
    This function is used to demonstrate fuzzing.
    """
    import json
    return json.loads(json_str)


def demonstrate_operator() -> None:
    """Demonstrate the use of the BaseFeludaOperator class."""
    log.info("=== Demonstrating BaseFeludaOperator ===")
    
    # Create an operator
    operator = ImageProcessor(parameters={
        "resize_width": 300,
        "resize_height": 300,
        "normalize": True,
    })
    
    # Create a media content object
    media_content = MediaContent(
        metadata=MediaMetadata(
            media_id="image1",
            media_type=MediaType.IMAGE,
            source="demo",
        ),
        content_data="simulated_image_data",
    )
    
    # Process the image
    result = operator.run(media_content)
    
    log.info(f"Operator result: {result}")


def demonstrate_resilience() -> None:
    """Demonstrate the resilience features."""
    log.info("=== Demonstrating Resilience ===")
    
    # Demonstrate circuit breaker
    log.info("--- Circuit Breaker ---")
    
    for i in range(5):
        try:
            result = call_external_service()
            log.info(f"Service call succeeded: {result}")
        except Exception as e:
            log.error(f"Service call failed: {e}")


def demonstrate_observability() -> None:
    """Demonstrate the observability features."""
    log.info("=== Demonstrating Observability ===")
    
    # Process some data to generate metrics and traces
    data = {"key1": "value1", "key2": "value2"}
    result = process_data(data)
    
    log.info(f"Processed data: {result}")
    
    # In a real application, you would use a tool like Jaeger or Zipkin
    # to view the traces and Prometheus to view the metrics


def demonstrate_chaos_testing() -> None:
    """Demonstrate chaos testing."""
    log.info("=== Demonstrating Chaos Testing ===")
    
    # Use the chaos context
    with chaos_context(
        enabled=True,
        failure_probability=0.5,
        enabled_failure_modes=["exception", "delay"],
        max_delay_ms=1000,
    ):
        # Process some data with chaos
        for i in range(5):
            try:
                data = {"key": f"value{i}"}
                result = process_data(data)
                log.info(f"Processed data: {result}")
            except Exception as e:
                log.error(f"Processing failed: {e}")


def demonstrate_fuzzing() -> None:
    """Demonstrate fuzzing."""
    log.info("=== Demonstrating Fuzzing ===")
    
    # Create a fuzzing configuration
    config = FuzzingConfig(
        strategy=FuzzingStrategy.GRAMMAR_BASED,
        grammar=JSON_GRAMMAR,
        min_length=10,
        max_length=100,
        seed=42,
    )
    
    # Fuzz the JSON parser
    results = fuzz_function(parse_json, config, iterations=10)
    
    log.info(f"Fuzzing results: {results}")


def demonstrate_metamorphic_testing() -> None:
    """Demonstrate metamorphic testing."""
    log.info("=== Demonstrating Metamorphic Testing ===")
    
    # Create metamorphic relations
    relations = [
        # Sorting a list and then adding an element is the same as adding the element and then sorting
        EqualityRelation(
            transformation=lambda lst: sorted(lst + [100]),
        ),
        # Sorting a reversed list is the same as sorting the original list
        EqualityRelation(
            transformation=lambda lst: sorted(list(reversed(lst))),
        ),
    ]
    
    # Run metamorphic tests
    results = run_metamorphic_tests(sort_list, [1, 3, 2, 4], relations)
    
    log.info(f"Metamorphic testing results: {results}")


def demonstrate_ml_tuning() -> None:
    """Demonstrate ML-driven tuning."""
    log.info("=== Demonstrating ML-Driven Tuning ===")
    
    # Create parameters
    params = [
        Parameter(
            name="x",
            parameter_type=ParameterType.CONTINUOUS,
            min_value=-5.0,
            max_value=5.0,
        ),
        Parameter(
            name="y",
            parameter_type=ParameterType.CONTINUOUS,
            min_value=-5.0,
            max_value=5.0,
        ),
    ]
    
    # Create a config
    config = OptimizationConfig(
        algorithm=OptimizationAlgorithm.RANDOM_SEARCH,
        parameters=params,
        max_iterations=20,
        random_seed=42,
    )
    
    # Define an objective function (minimize x^2 + y^2)
    def objective_function(params):
        return -((params["x"] ** 2) + (params["y"] ** 2))
    
    # Optimize
    best_params = optimize_parameters(objective_function, config)
    
    log.info(f"Best parameters: {best_params}")


def demonstrate_self_healing() -> None:
    """Demonstrate self-healing capabilities."""
    log.info("=== Demonstrating Self-Healing ===")
    
    # Create a self-healing system
    system = SelfHealingSystem()
    
    # Create a health check
    def check_function():
        # Simulate a health check
        if np.random.random() < 0.3:
            return HealthStatus.DEGRADED
        elif np.random.random() < 0.1:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.HEALTHY
    
    healing_actions = {
        HealthStatus.DEGRADED: [
            HealingAction(
                strategy=HealingStrategy.RETRY,
                params={"max_retries": 3, "retry_delay": 1.0},
            ),
        ],
        HealthStatus.UNHEALTHY: [
            HealingAction(
                strategy=HealingStrategy.RESTART,
                params={"restart_function": lambda: log.info("Restarting service...")},
            ),
        ],
    }
    
    health_check = HealthCheck(
        name="service",
        check_function=check_function,
        healing_actions=healing_actions,
        check_interval=1.0,  # Short interval for demo purposes
    )
    
    # Add the health check
    system.add_health_check(health_check)
    
    # Check health and heal
    for i in range(5):
        log.info(f"Iteration {i+1}")
        
        # Check health
        health = system.check_health()
        log.info(f"Health status: {health}")
        
        # Heal if needed
        healing_results = system.heal()
        if healing_results["service"]:
            log.info(f"Healing actions taken: {healing_results}")
        
        time.sleep(1)


def demonstrate_hardware_integration() -> None:
    """Demonstrate hardware integration."""
    log.info("=== Demonstrating Hardware Integration ===")
    
    # Create an FPGA configuration
    config = FPGAConfig(
        vendor=FPGAVendor.XILINX,
        family=FPGAFamily.XILINX_ULTRASCALE,
        hdl_language=HDLLanguage.VERILOG,
        part_number="xcvu9p-flgb2104-2-i",
        clock_frequency_mhz=100.0,
    )
    
    # Create an FPGA interface
    interface = FPGAInterface(config)
    
    # Generate HDL code for a simple function
    def add(a, b):
        return a + b
    
    hdl_code = interface.generate_hdl(
        function=add,
        input_types=[int, int],
        output_type=int,
        module_name="adder",
    )
    
    log.info(f"Generated HDL code:\n{hdl_code[:200]}...")  # Show first 200 chars


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Demonstrate Feluda's advanced features")
    parser.add_argument(
        "--feature",
        type=str,
        choices=[
            "operator",
            "resilience",
            "observability",
            "chaos",
            "fuzzing",
            "metamorphic",
            "ml_tuning",
            "self_healing",
            "hardware",
            "all",
        ],
        default="all",
        help="The feature to demonstrate",
    )
    
    args = parser.parse_args()
    
    if args.feature == "operator" or args.feature == "all":
        demonstrate_operator()
    
    if args.feature == "resilience" or args.feature == "all":
        demonstrate_resilience()
    
    if args.feature == "observability" or args.feature == "all":
        demonstrate_observability()
    
    if args.feature == "chaos" or args.feature == "all":
        demonstrate_chaos_testing()
    
    if args.feature == "fuzzing" or args.feature == "all":
        demonstrate_fuzzing()
    
    if args.feature == "metamorphic" or args.feature == "all":
        demonstrate_metamorphic_testing()
    
    if args.feature == "ml_tuning" or args.feature == "all":
        demonstrate_ml_tuning()
    
    if args.feature == "self_healing" or args.feature == "all":
        demonstrate_self_healing()
    
    if args.feature == "hardware" or args.feature == "all":
        demonstrate_hardware_integration()


if __name__ == "__main__":
    main()
