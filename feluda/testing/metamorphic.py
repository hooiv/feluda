"""
Metamorphic Testing Module

This module provides tools for metamorphic testing in Feluda.
Metamorphic testing involves testing relations between multiple executions of a function.
"""

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class MetamorphicRelation(str, Enum):
    """Enum for metamorphic relations."""
    
    EQUALITY = "equality"
    ADDITION = "addition"
    MULTIPLICATION = "multiplication"
    INVERSE = "inverse"
    PERMUTATION = "permutation"
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    NEGATION = "negation"
    IDEMPOTENCE = "idempotence"


class MetamorphicTest:
    """
    Base class for metamorphic tests.
    
    This class defines the interface for metamorphic tests.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        relation: MetamorphicRelation,
        tolerance: float = 1e-6,
    ):
        """
        Initialize a MetamorphicTest.
        
        Args:
            relation: The metamorphic relation to test.
            tolerance: The tolerance for floating-point comparisons.
        """
        self.relation = relation
        self.tolerance = tolerance
    
    def transform_input(self, input_data: Any) -> Any:
        """
        Transform the input data according to the metamorphic relation.
        
        Args:
            input_data: The input data to transform.
            
        Returns:
            The transformed input data.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement transform_input")
    
    def verify_relation(self, original_output: Any, transformed_output: Any) -> bool:
        """
        Verify that the metamorphic relation holds between the outputs.
        
        Args:
            original_output: The output from the original input.
            transformed_output: The output from the transformed input.
            
        Returns:
            True if the relation holds, False otherwise.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement verify_relation")


class EqualityRelation(MetamorphicTest):
    """
    Metamorphic test for equality relations.
    
    This test verifies that certain transformations of the input
    produce the same output.
    """
    
    def __init__(
        self,
        transformation: Callable[[Any], Any],
        tolerance: float = 1e-6,
    ):
        """
        Initialize an EqualityRelation.
        
        Args:
            transformation: The transformation to apply to the input.
            tolerance: The tolerance for floating-point comparisons.
        """
        super().__init__(MetamorphicRelation.EQUALITY, tolerance)
        self.transformation = transformation
    
    def transform_input(self, input_data: Any) -> Any:
        """
        Transform the input data using the specified transformation.
        
        Args:
            input_data: The input data to transform.
            
        Returns:
            The transformed input data.
        """
        return self.transformation(input_data)
    
    def verify_relation(self, original_output: Any, transformed_output: Any) -> bool:
        """
        Verify that the outputs are equal.
        
        Args:
            original_output: The output from the original input.
            transformed_output: The output from the transformed input.
            
        Returns:
            True if the outputs are equal, False otherwise.
        """
        if isinstance(original_output, (int, float)) and isinstance(transformed_output, (int, float)):
            return abs(original_output - transformed_output) <= self.tolerance
        elif isinstance(original_output, np.ndarray) and isinstance(transformed_output, np.ndarray):
            return np.allclose(original_output, transformed_output, rtol=self.tolerance)
        else:
            return original_output == transformed_output


class AdditionRelation(MetamorphicTest):
    """
    Metamorphic test for addition relations.
    
    This test verifies that adding a value to the input
    adds a corresponding value to the output.
    """
    
    def __init__(
        self,
        input_delta: Any,
        output_delta: Any,
        tolerance: float = 1e-6,
    ):
        """
        Initialize an AdditionRelation.
        
        Args:
            input_delta: The value to add to the input.
            output_delta: The expected value to add to the output.
            tolerance: The tolerance for floating-point comparisons.
        """
        super().__init__(MetamorphicRelation.ADDITION, tolerance)
        self.input_delta = input_delta
        self.output_delta = output_delta
    
    def transform_input(self, input_data: Any) -> Any:
        """
        Add the input delta to the input data.
        
        Args:
            input_data: The input data to transform.
            
        Returns:
            The transformed input data.
        """
        if isinstance(input_data, (int, float)) and isinstance(self.input_delta, (int, float)):
            return input_data + self.input_delta
        elif isinstance(input_data, np.ndarray) and isinstance(self.input_delta, np.ndarray):
            return input_data + self.input_delta
        elif isinstance(input_data, list) and isinstance(self.input_delta, list):
            return [x + y for x, y in zip(input_data, self.input_delta)]
        else:
            raise TypeError(f"Unsupported input types: {type(input_data)} and {type(self.input_delta)}")
    
    def verify_relation(self, original_output: Any, transformed_output: Any) -> bool:
        """
        Verify that the transformed output equals the original output plus the output delta.
        
        Args:
            original_output: The output from the original input.
            transformed_output: The output from the transformed input.
            
        Returns:
            True if the relation holds, False otherwise.
        """
        if isinstance(original_output, (int, float)) and isinstance(transformed_output, (int, float)) and isinstance(self.output_delta, (int, float)):
            return abs((original_output + self.output_delta) - transformed_output) <= self.tolerance
        elif isinstance(original_output, np.ndarray) and isinstance(transformed_output, np.ndarray) and isinstance(self.output_delta, np.ndarray):
            return np.allclose(original_output + self.output_delta, transformed_output, rtol=self.tolerance)
        elif isinstance(original_output, list) and isinstance(transformed_output, list) and isinstance(self.output_delta, list):
            expected = [x + y for x, y in zip(original_output, self.output_delta)]
            return all(abs(e - t) <= self.tolerance for e, t in zip(expected, transformed_output))
        else:
            raise TypeError(f"Unsupported output types: {type(original_output)}, {type(transformed_output)}, and {type(self.output_delta)}")


class MultiplicationRelation(MetamorphicTest):
    """
    Metamorphic test for multiplication relations.
    
    This test verifies that multiplying the input by a factor
    multiplies the output by a corresponding factor.
    """
    
    def __init__(
        self,
        input_factor: Any,
        output_factor: Any,
        tolerance: float = 1e-6,
    ):
        """
        Initialize a MultiplicationRelation.
        
        Args:
            input_factor: The factor to multiply the input by.
            output_factor: The expected factor to multiply the output by.
            tolerance: The tolerance for floating-point comparisons.
        """
        super().__init__(MetamorphicRelation.MULTIPLICATION, tolerance)
        self.input_factor = input_factor
        self.output_factor = output_factor
    
    def transform_input(self, input_data: Any) -> Any:
        """
        Multiply the input data by the input factor.
        
        Args:
            input_data: The input data to transform.
            
        Returns:
            The transformed input data.
        """
        if isinstance(input_data, (int, float)) and isinstance(self.input_factor, (int, float)):
            return input_data * self.input_factor
        elif isinstance(input_data, np.ndarray) and isinstance(self.input_factor, (int, float, np.ndarray)):
            return input_data * self.input_factor
        elif isinstance(input_data, list) and isinstance(self.input_factor, (int, float)):
            return [x * self.input_factor for x in input_data]
        else:
            raise TypeError(f"Unsupported input types: {type(input_data)} and {type(self.input_factor)}")
    
    def verify_relation(self, original_output: Any, transformed_output: Any) -> bool:
        """
        Verify that the transformed output equals the original output times the output factor.
        
        Args:
            original_output: The output from the original input.
            transformed_output: The output from the transformed input.
            
        Returns:
            True if the relation holds, False otherwise.
        """
        if isinstance(original_output, (int, float)) and isinstance(transformed_output, (int, float)) and isinstance(self.output_factor, (int, float)):
            return abs((original_output * self.output_factor) - transformed_output) <= self.tolerance
        elif isinstance(original_output, np.ndarray) and isinstance(transformed_output, np.ndarray) and isinstance(self.output_factor, (int, float, np.ndarray)):
            return np.allclose(original_output * self.output_factor, transformed_output, rtol=self.tolerance)
        elif isinstance(original_output, list) and isinstance(transformed_output, list) and isinstance(self.output_factor, (int, float)):
            expected = [x * self.output_factor for x in original_output]
            return all(abs(e - t) <= self.tolerance for e, t in zip(expected, transformed_output))
        else:
            raise TypeError(f"Unsupported output types: {type(original_output)}, {type(transformed_output)}, and {type(self.output_factor)}")


class InverseRelation(MetamorphicTest):
    """
    Metamorphic test for inverse relations.
    
    This test verifies that applying a function and then its inverse
    returns the original input.
    """
    
    def __init__(
        self,
        function: Callable[[Any], Any],
        inverse_function: Callable[[Any], Any],
        tolerance: float = 1e-6,
    ):
        """
        Initialize an InverseRelation.
        
        Args:
            function: The function to apply.
            inverse_function: The inverse of the function.
            tolerance: The tolerance for floating-point comparisons.
        """
        super().__init__(MetamorphicRelation.INVERSE, tolerance)
        self.function = function
        self.inverse_function = inverse_function
    
    def transform_input(self, input_data: Any) -> Any:
        """
        Apply the function to the input data.
        
        Args:
            input_data: The input data to transform.
            
        Returns:
            The transformed input data.
        """
        return self.function(input_data)
    
    def verify_relation(self, original_output: Any, transformed_output: Any) -> bool:
        """
        Verify that applying the inverse function to the transformed output
        returns the original output.
        
        Args:
            original_output: The output from the original input.
            transformed_output: The output from the transformed input.
            
        Returns:
            True if the relation holds, False otherwise.
        """
        inverse_output = self.inverse_function(transformed_output)
        
        if isinstance(original_output, (int, float)) and isinstance(inverse_output, (int, float)):
            return abs(original_output - inverse_output) <= self.tolerance
        elif isinstance(original_output, np.ndarray) and isinstance(inverse_output, np.ndarray):
            return np.allclose(original_output, inverse_output, rtol=self.tolerance)
        else:
            return original_output == inverse_output


def run_metamorphic_test(
    func: Callable[[Any], Any],
    input_data: Any,
    test: MetamorphicTest,
) -> Dict[str, Any]:
    """
    Run a metamorphic test on a function.
    
    Args:
        func: The function to test.
        input_data: The input data to use for testing.
        test: The metamorphic test to run.
        
    Returns:
        A dictionary with the test results.
    """
    # Run the function on the original input
    original_output = func(input_data)
    
    # Transform the input
    transformed_input = test.transform_input(input_data)
    
    # Run the function on the transformed input
    transformed_output = func(transformed_input)
    
    # Verify the relation
    relation_holds = test.verify_relation(original_output, transformed_output)
    
    # Return the results
    return {
        "relation": test.relation,
        "original_input": input_data,
        "transformed_input": transformed_input,
        "original_output": original_output,
        "transformed_output": transformed_output,
        "relation_holds": relation_holds,
    }


def run_metamorphic_tests(
    func: Callable[[Any], Any],
    input_data: Any,
    tests: List[MetamorphicTest],
) -> Dict[str, Any]:
    """
    Run multiple metamorphic tests on a function.
    
    Args:
        func: The function to test.
        input_data: The input data to use for testing.
        tests: The metamorphic tests to run.
        
    Returns:
        A dictionary with the test results.
    """
    results = {
        "function": func.__name__,
        "input_data": input_data,
        "tests": [],
        "passed": 0,
        "failed": 0,
    }
    
    for test in tests:
        test_result = run_metamorphic_test(func, input_data, test)
        results["tests"].append(test_result)
        
        if test_result["relation_holds"]:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    return results
