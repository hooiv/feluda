"""
Tests for the BaseFeludaOperator class.
"""

import unittest
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from feluda.base_operator import BaseFeludaOperator
from feluda.exceptions import (
    OperatorContractError,
    OperatorExecutionError,
    OperatorInitializationError,
    OperatorValidationError,
)


class TestParameters(BaseModel):
    """Test parameters model for the TestOperator."""
    
    param1: str = Field(default="default_value")
    param2: int = Field(default=42)
    param3: Optional[List[str]] = Field(default=None)


class TestOperator(BaseFeludaOperator[str, Dict[str, Any], TestParameters]):
    """Test operator implementation for testing."""
    
    name = "TestOperator"
    description = "A test operator for unit testing"
    version = "1.0.0"
    parameters_model = TestParameters
    
    def _initialize(self) -> None:
        """Initialize the operator."""
        self.initialized = True
    
    def _execute(self, input_data: str) -> Dict[str, Any]:
        """Execute the operator."""
        if input_data == "fail":
            raise ValueError("Execution failed")
        
        return {
            "input": input_data,
            "param1": self.parameters.param1,
            "param2": self.parameters.param2,
            "processed": True
        }
    
    def _validate_input(self, input_data: str) -> bool:
        """Validate the input data."""
        return input_data != "invalid"


class TestBaseFeludaOperator(unittest.TestCase):
    """Tests for the BaseFeludaOperator class."""
    
    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters."""
        operator = TestOperator()
        
        self.assertTrue(hasattr(operator, "initialized"))
        self.assertTrue(operator.initialized)
        self.assertEqual(operator.parameters.param1, "default_value")
        self.assertEqual(operator.parameters.param2, 42)
        self.assertIsNone(operator.parameters.param3)
    
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        operator = TestOperator(parameters={
            "param1": "custom_value",
            "param2": 100,
            "param3": ["item1", "item2"]
        })
        
        self.assertTrue(operator.initialized)
        self.assertEqual(operator.parameters.param1, "custom_value")
        self.assertEqual(operator.parameters.param2, 100)
        self.assertEqual(operator.parameters.param3, ["item1", "item2"])
    
    def test_initialization_with_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        with self.assertRaises(OperatorValidationError):
            TestOperator(parameters={
                "param1": "custom_value",
                "param2": "not_an_integer"  # This should be an integer
            })
    
    def test_successful_execution(self):
        """Test successful execution."""
        operator = TestOperator()
        result = operator.run("test_input")
        
        self.assertEqual(result["input"], "test_input")
        self.assertEqual(result["param1"], "default_value")
        self.assertEqual(result["param2"], 42)
        self.assertTrue(result["processed"])
    
    def test_execution_with_invalid_input(self):
        """Test execution with invalid input."""
        operator = TestOperator()
        
        with self.assertRaises(OperatorContractError) as context:
            operator.run("invalid")
        
        self.assertEqual(context.exception.contract_type, "pre")
    
    def test_execution_failure(self):
        """Test execution failure."""
        operator = TestOperator()
        
        with self.assertRaises(OperatorExecutionError) as context:
            operator.run("fail")
        
        self.assertEqual(context.exception.operator_type, "TestOperator")
        self.assertEqual(context.exception.input_data, "fail")
    
    def test_get_info(self):
        """Test get_info method."""
        operator = TestOperator()
        info = operator.get_info()
        
        self.assertEqual(info["name"], "TestOperator")
        self.assertEqual(info["description"], "A test operator for unit testing")
        self.assertEqual(info["version"], "1.0.0")
        self.assertIn("parameters", info)
        self.assertIn("input_type", info)
        self.assertIn("output_type", info)


class BrokenInitOperator(BaseFeludaOperator[str, Dict[str, Any], TestParameters]):
    """Test operator with broken initialization."""
    
    name = "BrokenInitOperator"
    description = "An operator that fails during initialization"
    version = "1.0.0"
    parameters_model = TestParameters
    
    def _initialize(self) -> None:
        """Initialize the operator with an error."""
        raise RuntimeError("Initialization failed")
    
    def _execute(self, input_data: str) -> Dict[str, Any]:
        """Execute the operator."""
        return {"result": "This should not be reached"}


class TestBrokenOperator(unittest.TestCase):
    """Tests for operators with broken initialization."""
    
    def test_initialization_failure(self):
        """Test initialization failure."""
        with self.assertRaises(OperatorInitializationError) as context:
            BrokenInitOperator()
        
        self.assertEqual(context.exception.operator_type, "BrokenInitOperator")
        self.assertIn("cause", context.exception.details)
        self.assertEqual(context.exception.details["cause"], "Initialization failed")


if __name__ == "__main__":
    unittest.main()
