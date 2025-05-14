"""
Autonomous QA Agent Module

This module provides an autonomous QA agent for generating and running tests.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import requests

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class QAAgent:
    """
    Autonomous QA agent for generating and running tests.
    
    This class implements an autonomous QA agent that can analyze code changes,
    generate tests, and report findings.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str,
        repo_path: str,
        **kwargs: Any,
    ):
        """
        Initialize a QA agent.
        
        Args:
            model: The name of the language model to use.
            api_key: The API key for the language model service.
            api_url: The URL of the language model service API.
            repo_path: The path to the repository to analyze.
            **kwargs: Additional agent-specific parameters.
        """
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.repo_path = repo_path
        self.system_prompt = (
            "You are an expert QA engineer. Your task is to analyze code changes, "
            "generate tests, and report findings. You should focus on edge cases, "
            "potential bugs, and areas where the code might not work as expected."
        )
    
    def analyze_pr(self, pr_diff: str) -> Dict[str, Any]:
        """
        Analyze a pull request diff.
        
        Args:
            pr_diff: The diff of the pull request to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        # Call the language model to analyze the PR
        prompt = f"Analyze the following pull request diff and identify potential issues, edge cases, and areas that need testing:\n\n{pr_diff}"
        analysis = self._call_llm(prompt)
        
        # Parse the analysis
        try:
            # Try to parse the analysis as JSON
            result = json.loads(analysis)
        except json.JSONDecodeError:
            # If parsing fails, return the raw analysis
            result = {"raw_analysis": analysis}
        
        return result
    
    def generate_tests(self, file_path: str, code: str) -> List[Dict[str, Any]]:
        """
        Generate tests for a file.
        
        Args:
            file_path: The path to the file to generate tests for.
            code: The code to generate tests for.
            
        Returns:
            A list of test cases.
        """
        # Call the language model to generate tests
        prompt = f"Generate pytest test cases for the following code:\n\nFile: {file_path}\n\nCode:\n{code}\n\nReturn the test cases as a JSON array of objects, where each object has 'name', 'description', and 'code' fields."
        tests_json = self._call_llm(prompt)
        
        # Parse the tests
        try:
            tests = json.loads(tests_json)
            if not isinstance(tests, list):
                tests = [{"name": "default", "description": "Default test", "code": tests_json}]
        except json.JSONDecodeError:
            tests = [{"name": "default", "description": "Default test", "code": tests_json}]
        
        return tests
    
    def run_tests(self, tests: List[Dict[str, Any]], file_path: str) -> Dict[str, Any]:
        """
        Run tests for a file.
        
        Args:
            tests: The tests to run.
            file_path: The path to the file to run tests for.
            
        Returns:
            A dictionary containing the test results.
        """
        results = {
            "total": len(tests),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "test_results": [],
        }
        
        # Create a temporary directory for the tests
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the tests to files
            for i, test in enumerate(tests):
                test_file_path = os.path.join(temp_dir, f"test_{i}.py")
                with open(test_file_path, "w") as f:
                    f.write(test["code"])
                
                # Run the test
                try:
                    result = subprocess.run(
                        ["pytest", test_file_path, "-v"],
                        capture_output=True,
                        text=True,
                        cwd=self.repo_path,
                    )
                    
                    # Parse the result
                    if result.returncode == 0:
                        status = "passed"
                        results["passed"] += 1
                    else:
                        status = "failed"
                        results["failed"] += 1
                    
                    test_result = {
                        "name": test["name"],
                        "description": test["description"],
                        "status": status,
                        "output": result.stdout,
                        "error": result.stderr,
                    }
                    
                    results["test_results"].append(test_result)
                    
                except Exception as e:
                    results["errors"] += 1
                    test_result = {
                        "name": test["name"],
                        "description": test["description"],
                        "status": "error",
                        "output": "",
                        "error": str(e),
                    }
                    
                    results["test_results"].append(test_result)
        
        return results
    
    def generate_hypothesis_strategies(self, file_path: str, code: str) -> List[Dict[str, Any]]:
        """
        Generate Hypothesis strategies for property-based testing.
        
        Args:
            file_path: The path to the file to generate strategies for.
            code: The code to generate strategies for.
            
        Returns:
            A list of Hypothesis strategies.
        """
        # Call the language model to generate Hypothesis strategies
        prompt = f"Generate Hypothesis strategies for property-based testing of the following code:\n\nFile: {file_path}\n\nCode:\n{code}\n\nReturn the strategies as a JSON array of objects, where each object has 'name', 'description', and 'code' fields."
        strategies_json = self._call_llm(prompt)
        
        # Parse the strategies
        try:
            strategies = json.loads(strategies_json)
            if not isinstance(strategies, list):
                strategies = [{"name": "default", "description": "Default strategy", "code": strategies_json}]
        except json.JSONDecodeError:
            strategies = [{"name": "default", "description": "Default strategy", "code": strategies_json}]
        
        return strategies
    
    def analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze test results.
        
        Args:
            test_results: The test results to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        # Call the language model to analyze the test results
        prompt = f"Analyze the following test results and provide insights and recommendations:\n\n{json.dumps(test_results, indent=2)}"
        analysis = self._call_llm(prompt)
        
        # Parse the analysis
        try:
            # Try to parse the analysis as JSON
            result = json.loads(analysis)
        except json.JSONDecodeError:
            # If parsing fails, return the raw analysis
            result = {"raw_analysis": analysis}
        
        return result
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the language model.
        
        Args:
            prompt: The prompt to send to the language model.
            
        Returns:
            The response from the language model.
            
        Raises:
            Exception: If the API call fails.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")


class PRAnalyzer:
    """
    Analyzer for pull requests.
    
    This class analyzes pull requests and generates test cases.
    """
    
    def __init__(
        self,
        qa_agent: QAAgent,
        repo_path: str,
        **kwargs: Any,
    ):
        """
        Initialize a PR analyzer.
        
        Args:
            qa_agent: The QA agent to use for analysis.
            repo_path: The path to the repository to analyze.
            **kwargs: Additional analyzer-specific parameters.
        """
        self.qa_agent = qa_agent
        self.repo_path = repo_path
    
    def analyze_pr(self, pr_number: int) -> Dict[str, Any]:
        """
        Analyze a pull request.
        
        Args:
            pr_number: The number of the pull request to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        # Get the PR diff
        diff = self._get_pr_diff(pr_number)
        
        # Analyze the PR
        analysis = self.qa_agent.analyze_pr(diff)
        
        # Get the changed files
        changed_files = self._get_changed_files(pr_number)
        
        # Generate tests for each changed file
        tests = {}
        for file_path in changed_files:
            if file_path.endswith(".py"):
                code = self._get_file_content(file_path)
                file_tests = self.qa_agent.generate_tests(file_path, code)
                tests[file_path] = file_tests
        
        # Run the tests
        test_results = {}
        for file_path, file_tests in tests.items():
            test_results[file_path] = self.qa_agent.run_tests(file_tests, file_path)
        
        # Analyze the test results
        test_analysis = self.qa_agent.analyze_test_results(test_results)
        
        # Return the results
        return {
            "pr_number": pr_number,
            "analysis": analysis,
            "tests": tests,
            "test_results": test_results,
            "test_analysis": test_analysis,
        }
    
    def _get_pr_diff(self, pr_number: int) -> str:
        """
        Get the diff of a pull request.
        
        Args:
            pr_number: The number of the pull request.
            
        Returns:
            The diff of the pull request.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use the GitHub API or git commands
        return f"Diff for PR #{pr_number}"
    
    def _get_changed_files(self, pr_number: int) -> List[str]:
        """
        Get the list of files changed in a pull request.
        
        Args:
            pr_number: The number of the pull request.
            
        Returns:
            The list of changed files.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use the GitHub API or git commands
        return [f"file_{i}.py" for i in range(3)]
    
    def _get_file_content(self, file_path: str) -> str:
        """
        Get the content of a file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            The content of the file.
        """
        # This is a placeholder implementation
        # In a real implementation, this would read the file from the repository
        return f"Content of {file_path}"
