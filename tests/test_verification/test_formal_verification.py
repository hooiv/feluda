"""
Unit tests for the formal verification module.
"""

import unittest
from unittest import mock

import deal
import pytest

from feluda.verification.formal_verification import (
    CrosshairVerifier,
    DealVerifier,
    FormalVerifier,
    VerificationReport,
    VerificationResult,
    create_verifier,
    has_contracts,
    verify_function,
)


# Define some functions to test
def function_without_contracts(x):
    """A function without contracts."""
    return x + 1


@deal.pre(lambda x: x > 0)
@deal.post(lambda result: result > 1)
def function_with_contracts(x):
    """A function with contracts."""
    return x + 1


@deal.pre(lambda x: x > 0)
@deal.post(lambda result: result > 100)  # This will fail for small x
def function_with_failing_contracts(x):
    """A function with failing contracts."""
    return x + 1


class TestFormalVerification(unittest.TestCase):
    """Test cases for the formal verification module."""
    
    def test_verification_report(self):
        """Test the VerificationReport class."""
        # Create a report
        report = VerificationReport(
            function_name="test_function",
            result=VerificationResult.VERIFIED,
            execution_time=1.0,
        )
        
        # Check the attributes
        self.assertEqual(report.function_name, "test_function")
        self.assertEqual(report.result, VerificationResult.VERIFIED)
        self.assertEqual(report.execution_time, 1.0)
        self.assertIsNone(report.counterexample)
        self.assertIsNone(report.error_message)
        
        # Test to_dict
        report_dict = report.to_dict()
        self.assertEqual(report_dict["function_name"], "test_function")
        self.assertEqual(report_dict["result"], VerificationResult.VERIFIED)
        self.assertEqual(report_dict["execution_time"], 1.0)
        self.assertIsNone(report_dict["counterexample"])
        self.assertIsNone(report_dict["error_message"])
        
        # Test __str__
        report_str = str(report)
        self.assertIn("test_function", report_str)
        self.assertIn("verified", report_str)
        self.assertIn("1.0", report_str)
    
    def test_deal_verifier(self):
        """Test the DealVerifier class."""
        # Create a verifier
        verifier = DealVerifier(timeout=1.0)
        
        # Verify a function without contracts
        report = verifier.verify(function_without_contracts)
        self.assertEqual(report.result, VerificationResult.UNKNOWN)
        self.assertIn("No contracts found", report.error_message)
        
        # Verify a function with contracts
        report = verifier.verify(function_with_contracts)
        self.assertEqual(report.result, VerificationResult.VERIFIED)
        
        # Verify a function with failing contracts
        report = verifier.verify(function_with_failing_contracts)
        self.assertEqual(report.result, VerificationResult.FALSIFIED)
        self.assertIsNotNone(report.counterexample)
        self.assertIsNotNone(report.error_message)
    
    def test_crosshair_verifier(self):
        """Test the CrosshairVerifier class."""
        # Create a verifier
        verifier = CrosshairVerifier(timeout=1.0)
        
        # Mock the CrossHair library
        with mock.patch("feluda.verification.formal_verification.CrosshairVerifier.verify") as mock_verify:
            mock_verify.return_value = VerificationReport(
                function_name="test_function",
                result=VerificationResult.VERIFIED,
                execution_time=1.0,
            )
            
            # Verify a function
            report = verifier.verify(function_with_contracts)
            
            mock_verify.assert_called_once()
            self.assertEqual(report.result, VerificationResult.VERIFIED)
    
    def test_create_verifier(self):
        """Test the create_verifier function."""
        # Create a deal verifier
        verifier = create_verifier(verifier_type="deal", timeout=1.0)
        self.assertIsInstance(verifier, DealVerifier)
        self.assertEqual(verifier.timeout, 1.0)
        
        # Create a crosshair verifier
        verifier = create_verifier(verifier_type="crosshair", timeout=2.0)
        self.assertIsInstance(verifier, CrosshairVerifier)
        self.assertEqual(verifier.timeout, 2.0)
        
        # Test with an unsupported verifier type
        with self.assertRaises(ValueError):
            create_verifier(verifier_type="unsupported")
    
    def test_verify_function(self):
        """Test the verify_function function."""
        # Verify a function with deal
        with mock.patch("feluda.verification.formal_verification.create_verifier") as mock_create:
            mock_verifier = mock.MagicMock()
            mock_verifier.verify.return_value = VerificationReport(
                function_name="test_function",
                result=VerificationResult.VERIFIED,
                execution_time=1.0,
            )
            mock_create.return_value = mock_verifier
            
            report = verify_function(function_with_contracts, verifier_type="deal", timeout=1.0)
            
            mock_create.assert_called_once_with(verifier_type="deal", timeout=1.0)
            mock_verifier.verify.assert_called_once_with(function_with_contracts)
            
            self.assertEqual(report.result, VerificationResult.VERIFIED)
    
    def test_has_contracts(self):
        """Test the has_contracts function."""
        # Test a function without contracts
        self.assertFalse(has_contracts(function_without_contracts))
        
        # Test a function with contracts
        self.assertTrue(has_contracts(function_with_contracts))
        
        # Test a function with failing contracts
        self.assertTrue(has_contracts(function_with_failing_contracts))
        
        # Test with a non-callable
        with self.assertRaises(deal.PreContractError):
            has_contracts("not_callable")


if __name__ == "__main__":
    unittest.main()
