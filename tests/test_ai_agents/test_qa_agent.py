"""
Unit tests for the QA agent module.
"""

import unittest
from unittest import mock

import pytest

from feluda.ai_agents.qa_agent import (
    PRAnalyzer,
    QAAgent,
    QAConfig,
    TestCase,
    TestResult,
    TestSuite,
)


class TestQAAgent(unittest.TestCase):
    """Test cases for the QA agent module."""
    
    def test_test_case(self):
        """Test the TestCase class."""
        # Create a test case
        test_case = TestCase(
            name="test_function",
            description="Test the function",
            code="def test_function():\n    assert True",
            expected_result="Pass",
        )
        
        # Check the attributes
        self.assertEqual(test_case.name, "test_function")
        self.assertEqual(test_case.description, "Test the function")
        self.assertEqual(test_case.code, "def test_function():\n    assert True")
        self.assertEqual(test_case.expected_result, "Pass")
        
        # Test to_dict
        test_case_dict = test_case.to_dict()
        self.assertEqual(test_case_dict["name"], "test_function")
        self.assertEqual(test_case_dict["description"], "Test the function")
        self.assertEqual(test_case_dict["code"], "def test_function():\n    assert True")
        self.assertEqual(test_case_dict["expected_result"], "Pass")
        
        # Test from_dict
        test_case2 = TestCase.from_dict(test_case_dict)
        self.assertEqual(test_case2.name, "test_function")
        self.assertEqual(test_case2.description, "Test the function")
        self.assertEqual(test_case2.code, "def test_function():\n    assert True")
        self.assertEqual(test_case2.expected_result, "Pass")
    
    def test_test_result(self):
        """Test the TestResult class."""
        # Create a test result
        test_result = TestResult(
            test_case=TestCase(
                name="test_function",
                description="Test the function",
                code="def test_function():\n    assert True",
                expected_result="Pass",
            ),
            passed=True,
            output="Test passed",
            execution_time=0.1,
        )
        
        # Check the attributes
        self.assertEqual(test_result.test_case.name, "test_function")
        self.assertEqual(test_result.test_case.description, "Test the function")
        self.assertEqual(test_result.test_case.code, "def test_function():\n    assert True")
        self.assertEqual(test_result.test_case.expected_result, "Pass")
        self.assertTrue(test_result.passed)
        self.assertEqual(test_result.output, "Test passed")
        self.assertEqual(test_result.execution_time, 0.1)
        
        # Test to_dict
        test_result_dict = test_result.to_dict()
        self.assertEqual(test_result_dict["test_case"]["name"], "test_function")
        self.assertEqual(test_result_dict["test_case"]["description"], "Test the function")
        self.assertEqual(test_result_dict["test_case"]["code"], "def test_function():\n    assert True")
        self.assertEqual(test_result_dict["test_case"]["expected_result"], "Pass")
        self.assertTrue(test_result_dict["passed"])
        self.assertEqual(test_result_dict["output"], "Test passed")
        self.assertEqual(test_result_dict["execution_time"], 0.1)
        
        # Test from_dict
        test_result2 = TestResult.from_dict(test_result_dict)
        self.assertEqual(test_result2.test_case.name, "test_function")
        self.assertEqual(test_result2.test_case.description, "Test the function")
        self.assertEqual(test_result2.test_case.code, "def test_function():\n    assert True")
        self.assertEqual(test_result2.test_case.expected_result, "Pass")
        self.assertTrue(test_result2.passed)
        self.assertEqual(test_result2.output, "Test passed")
        self.assertEqual(test_result2.execution_time, 0.1)
    
    def test_test_suite(self):
        """Test the TestSuite class."""
        # Create a test suite
        test_suite = TestSuite(
            name="test_suite",
            description="Test suite",
            test_cases=[
                TestCase(
                    name="test_function1",
                    description="Test function 1",
                    code="def test_function1():\n    assert True",
                    expected_result="Pass",
                ),
                TestCase(
                    name="test_function2",
                    description="Test function 2",
                    code="def test_function2():\n    assert True",
                    expected_result="Pass",
                ),
            ],
        )
        
        # Check the attributes
        self.assertEqual(test_suite.name, "test_suite")
        self.assertEqual(test_suite.description, "Test suite")
        self.assertEqual(len(test_suite.test_cases), 2)
        self.assertEqual(test_suite.test_cases[0].name, "test_function1")
        self.assertEqual(test_suite.test_cases[1].name, "test_function2")
        
        # Test to_dict
        test_suite_dict = test_suite.to_dict()
        self.assertEqual(test_suite_dict["name"], "test_suite")
        self.assertEqual(test_suite_dict["description"], "Test suite")
        self.assertEqual(len(test_suite_dict["test_cases"]), 2)
        self.assertEqual(test_suite_dict["test_cases"][0]["name"], "test_function1")
        self.assertEqual(test_suite_dict["test_cases"][1]["name"], "test_function2")
        
        # Test from_dict
        test_suite2 = TestSuite.from_dict(test_suite_dict)
        self.assertEqual(test_suite2.name, "test_suite")
        self.assertEqual(test_suite2.description, "Test suite")
        self.assertEqual(len(test_suite2.test_cases), 2)
        self.assertEqual(test_suite2.test_cases[0].name, "test_function1")
        self.assertEqual(test_suite2.test_cases[1].name, "test_function2")
        
        # Test run
        with mock.patch("feluda.ai_agents.qa_agent.TestSuite._run_test_case") as mock_run_test_case:
            mock_run_test_case.return_value = TestResult(
                test_case=test_suite.test_cases[0],
                passed=True,
                output="Test passed",
                execution_time=0.1,
            )
            
            results = test_suite.run()
            
            self.assertEqual(mock_run_test_case.call_count, 2)
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0].passed)
            self.assertTrue(results[1].passed)
    
    def test_qa_config(self):
        """Test the QAConfig class."""
        # Create a config
        config = QAConfig(
            model="gpt-4",
            api_key="test_key",
            api_url="https://api.example.com",
            max_tokens=1000,
            temperature=0.7,
        )
        
        # Check the attributes
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.api_url, "https://api.example.com")
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.temperature, 0.7)
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["model"], "gpt-4")
        self.assertEqual(config_dict["api_key"], "test_key")
        self.assertEqual(config_dict["api_url"], "https://api.example.com")
        self.assertEqual(config_dict["max_tokens"], 1000)
        self.assertEqual(config_dict["temperature"], 0.7)
        
        # Test from_dict
        config2 = QAConfig.from_dict(config_dict)
        self.assertEqual(config2.model, "gpt-4")
        self.assertEqual(config2.api_key, "test_key")
        self.assertEqual(config2.api_url, "https://api.example.com")
        self.assertEqual(config2.max_tokens, 1000)
        self.assertEqual(config2.temperature, 0.7)
    
    def test_qa_agent(self):
        """Test the QAAgent class."""
        # Create an agent
        agent = QAAgent(
            model="gpt-4",
            api_key="test_key",
            api_url="https://api.example.com",
            repo_path="/path/to/repo",
        )
        
        # Check the attributes
        self.assertEqual(agent.config.model, "gpt-4")
        self.assertEqual(agent.config.api_key, "test_key")
        self.assertEqual(agent.config.api_url, "https://api.example.com")
        self.assertEqual(agent.repo_path, "/path/to/repo")
        
        # Test generate_test_suite
        with mock.patch("feluda.ai_agents.qa_agent.QAAgent._call_api") as mock_call_api:
            mock_call_api.return_value = {
                "name": "test_suite",
                "description": "Test suite",
                "test_cases": [
                    {
                        "name": "test_function",
                        "description": "Test function",
                        "code": "def test_function():\n    assert True",
                        "expected_result": "Pass",
                    },
                ],
            }
            
            test_suite = agent.generate_test_suite(
                code="def function():\n    return True",
                description="A function that returns True",
            )
            
            mock_call_api.assert_called_once()
            self.assertEqual(test_suite.name, "test_suite")
            self.assertEqual(test_suite.description, "Test suite")
            self.assertEqual(len(test_suite.test_cases), 1)
            self.assertEqual(test_suite.test_cases[0].name, "test_function")
    
    def test_pr_analyzer(self):
        """Test the PRAnalyzer class."""
        # Create an analyzer
        qa_agent = QAAgent(
            model="gpt-4",
            api_key="test_key",
            api_url="https://api.example.com",
            repo_path="/path/to/repo",
        )
        
        analyzer = PRAnalyzer(
            qa_agent=qa_agent,
            repo_path="/path/to/repo",
        )
        
        # Check the attributes
        self.assertEqual(analyzer.qa_agent, qa_agent)
        self.assertEqual(analyzer.repo_path, "/path/to/repo")
        
        # Test analyze_pr
        with mock.patch("feluda.ai_agents.qa_agent.PRAnalyzer._get_pr_files") as mock_get_pr_files:
            mock_get_pr_files.return_value = ["file1.py", "file2.py"]
            
            with mock.patch("feluda.ai_agents.qa_agent.PRAnalyzer._get_file_content") as mock_get_file_content:
                mock_get_file_content.return_value = "def function():\n    return True"
                
                with mock.patch("feluda.ai_agents.qa_agent.QAAgent.generate_test_suite") as mock_generate_test_suite:
                    mock_test_suite = mock.MagicMock()
                    mock_test_suite.run.return_value = [
                        TestResult(
                            test_case=TestCase(
                                name="test_function",
                                description="Test function",
                                code="def test_function():\n    assert True",
                                expected_result="Pass",
                            ),
                            passed=True,
                            output="Test passed",
                            execution_time=0.1,
                        ),
                    ]
                    mock_generate_test_suite.return_value = mock_test_suite
                    
                    analysis = analyzer.analyze_pr(pr_number=123)
                    
                    mock_get_pr_files.assert_called_once_with(123)
                    self.assertEqual(mock_get_file_content.call_count, 2)
                    self.assertEqual(mock_generate_test_suite.call_count, 2)
                    self.assertEqual(mock_test_suite.run.call_count, 2)
                    self.assertIsInstance(analysis, dict)
                    self.assertIn("files", analysis)
                    self.assertIn("test_suites", analysis)
                    self.assertIn("results", analysis)
                    self.assertIn("summary", analysis)


if __name__ == "__main__":
    unittest.main()
