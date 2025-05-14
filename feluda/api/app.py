"""
Web API for Feluda.

This module provides a web API for Feluda using FastAPI.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from feluda import Feluda
from feluda.ai_agents.agent_swarm import create_development_swarm
from feluda.ai_agents.qa_agent import PRAnalyzer, QAAgent
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
from feluda.observability import get_logger
from feluda.verification.formal_verification import (
    VerificationReport,
    VerificationResult,
    verify_function,
    verify_module,
)

log = get_logger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Feluda API",
    description="API for Feluda: A configurable engine for analysing multi-lingual and multi-modal content",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Feluda
feluda = None


# Define API models
class HealthResponse(BaseModel):
    """Health response model."""
    
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Feluda version")
    uptime: float = Field(..., description="Uptime in seconds")


class OperatorRequest(BaseModel):
    """Operator request model."""
    
    operator: str = Field(..., description="Operator name")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Operator parameters")


class OperatorResponse(BaseModel):
    """Operator response model."""
    
    operator: str = Field(..., description="Operator name")
    result: Dict[str, Any] = Field(..., description="Operator result")
    execution_time: float = Field(..., description="Execution time in seconds")


class VerificationRequest(BaseModel):
    """Verification request model."""
    
    module: Optional[str] = Field(None, description="Module to verify")
    function: Optional[str] = Field(None, description="Function to verify")
    verifier: str = Field("deal", description="Verifier to use")
    timeout: float = Field(10.0, description="Timeout in seconds")


class VerificationResponse(BaseModel):
    """Verification response model."""
    
    reports: List[Dict[str, Any]] = Field(..., description="Verification reports")
    execution_time: float = Field(..., description="Execution time in seconds")


class OptimizationRequest(BaseModel):
    """Optimization request model."""
    
    algorithm: str = Field(OptimizationAlgorithm.RANDOM_SEARCH, description="Optimization algorithm")
    parameters: List[Dict[str, Any]] = Field(..., description="Parameters to optimize")
    max_iterations: int = Field(100, description="Maximum number of iterations")
    random_seed: Optional[int] = Field(None, description="Random seed")
    algorithm_params: Dict[str, Any] = Field({}, description="Algorithm-specific parameters")


class OptimizationResponse(BaseModel):
    """Optimization response model."""
    
    best_params: Dict[str, Any] = Field(..., description="Best parameters")
    execution_time: float = Field(..., description="Execution time in seconds")


class HealingRequest(BaseModel):
    """Healing request model."""
    
    health_checks: List[Dict[str, Any]] = Field(..., description="Health checks")
    check_name: Optional[str] = Field(None, description="Name of the health check to run")


class HealingResponse(BaseModel):
    """Healing response model."""
    
    health: Dict[str, str] = Field(..., description="Health status")
    healing_results: Dict[str, List[Dict[str, Any]]] = Field(..., description="Healing results")
    execution_time: float = Field(..., description="Execution time in seconds")


class AgentSwarmRequest(BaseModel):
    """Agent swarm request model."""
    
    task: str = Field(..., description="Task for the swarm")
    code_context: Optional[str] = Field(None, description="Code context for the swarm")
    api_key: str = Field(..., description="API key for the language model")
    api_url: Optional[str] = Field(None, description="API URL for the language model")
    steps: int = Field(10, description="Number of steps to run")


class AgentSwarmResponse(BaseModel):
    """Agent swarm response model."""
    
    result: str = Field(..., description="Swarm result")
    messages: List[Dict[str, Any]] = Field(..., description="Swarm messages")
    execution_time: float = Field(..., description="Execution time in seconds")


class QARequest(BaseModel):
    """QA request model."""
    
    repo_path: str = Field(..., description="Path to the repository")
    pr_number: Optional[int] = Field(None, description="PR number to analyze")
    file_path: Optional[str] = Field(None, description="File to analyze")
    file_content: Optional[str] = Field(None, description="File content to analyze")
    api_key: str = Field(..., description="API key for the language model")
    api_url: Optional[str] = Field(None, description="API URL for the language model")


class QAResponse(BaseModel):
    """QA response model."""
    
    test_suite: Dict[str, Any] = Field(..., description="Test suite")
    results: List[Dict[str, Any]] = Field(..., description="Test results")
    summary: str = Field(..., description="Summary")
    execution_time: float = Field(..., description="Execution time in seconds")


# Define API endpoints
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Get the health status of the API.
    
    Returns:
        The health status.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": time.time() - start_time,
    }


@app.post("/operators", response_model=OperatorResponse)
async def run_operator(request: OperatorRequest):
    """
    Run an operator.
    
    Args:
        request: The operator request.
        
    Returns:
        The operator response.
    """
    start = time.time()
    
    try:
        # Check if Feluda is initialized
        if feluda is None:
            raise HTTPException(status_code=500, detail="Feluda is not initialized")
        
        # Get the operator
        operator = feluda.operators.get()[request.operator]
        
        # Set parameters
        if request.parameters:
            operator.parameters = request.parameters
        
        # Run the operator
        result = operator.run(request.input_data)
        
        return {
            "operator": request.operator,
            "result": result,
            "execution_time": time.time() - start,
        }
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Operator {request.operator} not found")
    
    except Exception as e:
        log.error(f"Error running operator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerificationResponse)
async def verify(request: VerificationRequest):
    """
    Verify Feluda components.
    
    Args:
        request: The verification request.
        
    Returns:
        The verification response.
    """
    start = time.time()
    
    try:
        reports = []
        
        if request.module:
            # Import the module
            module = __import__(request.module, fromlist=[""])
            
            # Verify the module
            module_reports = verify_module(
                module=module,
                verifier_type=request.verifier,
                timeout=request.timeout,
            )
            
            reports.extend([report.to_dict() for report in module_reports])
        
        elif request.function:
            # Import the function
            module_name, function_name = request.function.rsplit(".", 1)
            module = __import__(module_name, fromlist=[""])
            function = getattr(module, function_name)
            
            # Verify the function
            report = verify_function(
                function=function,
                verifier_type=request.verifier,
                timeout=request.timeout,
            )
            
            reports.append(report.to_dict())
        
        else:
            raise HTTPException(status_code=400, detail="Either module or function must be specified")
        
        return {
            "reports": reports,
            "execution_time": time.time() - start,
        }
    
    except Exception as e:
        log.error(f"Error verifying components: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    """
    Optimize parameters.
    
    Args:
        request: The optimization request.
        
    Returns:
        The optimization response.
    """
    start = time.time()
    
    try:
        # Create parameters
        parameters = []
        for param_data in request.parameters:
            parameter = Parameter(
                name=param_data["name"],
                parameter_type=param_data["type"],
                min_value=param_data.get("min_value"),
                max_value=param_data.get("max_value"),
                choices=param_data.get("choices"),
                default=param_data.get("default"),
            )
            parameters.append(parameter)
        
        # Create the optimization configuration
        config = OptimizationConfig(
            algorithm=request.algorithm,
            parameters=parameters,
            max_iterations=request.max_iterations,
            random_seed=request.random_seed,
            algorithm_params=request.algorithm_params,
        )
        
        # Define the objective function
        def objective_function(params):
            # In a real implementation, this would call the actual function
            # For now, we just return a dummy value
            log.info(f"Evaluating parameters: {params}")
            return sum(params.values())
        
        # Optimize the parameters
        best_params = optimize_parameters(objective_function, config)
        
        return {
            "best_params": best_params,
            "execution_time": time.time() - start,
        }
    
    except Exception as e:
        log.error(f"Error optimizing parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/heal", response_model=HealingResponse)
async def heal(request: HealingRequest):
    """
    Run self-healing.
    
    Args:
        request: The healing request.
        
    Returns:
        The healing response.
    """
    start = time.time()
    
    try:
        # Create the self-healing system
        system = SelfHealingSystem()
        
        # Add health checks
        for check_data in request.health_checks:
            # Create the check function
            check_function_code = check_data.get("check_function", "lambda: HealthStatus.HEALTHY")
            check_function = eval(check_function_code)
            
            # Create healing actions
            healing_actions = {}
            for status, actions in check_data.get("healing_actions", {}).items():
                healing_actions[status] = []
                for action_data in actions:
                    action = HealingAction(
                        strategy=action_data["strategy"],
                        params=action_data.get("params", {}),
                    )
                    healing_actions[status].append(action)
            
            # Create the health check
            health_check = HealthCheck(
                name=check_data["name"],
                check_function=check_function,
                healing_actions=healing_actions,
                check_interval=check_data.get("check_interval", 60.0),
            )
            
            # Add the health check
            system.add_health_check(health_check)
        
        # Check health
        health = system.check_health(request.check_name)
        
        # Heal
        healing_results = system.heal(request.check_name)
        
        return {
            "health": {k: v.value for k, v in health.items()},
            "healing_results": healing_results,
            "execution_time": time.time() - start,
        }
    
    except Exception as e:
        log.error(f"Error running self-healing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/swarm", response_model=AgentSwarmResponse)
async def agent_swarm(request: AgentSwarmRequest):
    """
    Run an agent swarm.
    
    Args:
        request: The agent swarm request.
        
    Returns:
        The agent swarm response.
    """
    start = time.time()
    
    try:
        # Create a development swarm
        swarm = create_development_swarm(
            task=request.task,
            code_context=request.code_context,
            api_key=request.api_key,
            api_url=request.api_url,
        )
        
        # Run the swarm
        result = swarm.run(steps=request.steps)
        
        return {
            "result": result["result"],
            "messages": result["messages"],
            "execution_time": time.time() - start,
        }
    
    except Exception as e:
        log.error(f"Error running agent swarm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/qa", response_model=QAResponse)
async def agent_qa(request: QARequest):
    """
    Run a QA agent.
    
    Args:
        request: The QA request.
        
    Returns:
        The QA response.
    """
    start = time.time()
    
    try:
        # Create a QA agent
        agent = QAAgent(
            model="gpt-4",
            api_key=request.api_key,
            api_url=request.api_url,
            repo_path=request.repo_path,
        )
        
        if request.pr_number:
            # Create a PR analyzer
            analyzer = PRAnalyzer(
                qa_agent=agent,
                repo_path=request.repo_path,
            )
            
            # Analyze the PR
            analysis = analyzer.analyze_pr(pr_number=request.pr_number)
            
            return {
                "test_suite": analysis["test_suites"],
                "results": analysis["results"],
                "summary": analysis["summary"],
                "execution_time": time.time() - start,
            }
        
        elif request.file_path:
            # Read the file
            with open(request.file_path, "r") as f:
                code = f.read()
            
            # Generate a test suite
            test_suite = agent.generate_test_suite(
                code=code,
                description=f"Test suite for {request.file_path}",
            )
            
            # Run the test suite
            results = test_suite.run()
            
            return {
                "test_suite": test_suite.to_dict(),
                "results": [r.to_dict() for r in results],
                "summary": f"{len([r for r in results if r.passed])}/{len(results)} tests passed",
                "execution_time": time.time() - start,
            }
        
        elif request.file_content:
            # Generate a test suite
            test_suite = agent.generate_test_suite(
                code=request.file_content,
                description="Test suite for the provided code",
            )
            
            # Run the test suite
            results = test_suite.run()
            
            return {
                "test_suite": test_suite.to_dict(),
                "results": [r.to_dict() for r in results],
                "summary": f"{len([r for r in results if r.passed])}/{len(results)} tests passed",
                "execution_time": time.time() - start,
            }
        
        else:
            raise HTTPException(status_code=400, detail="Either pr_number, file_path, or file_content must be specified")
    
    except Exception as e:
        log.error(f"Error running QA agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Initialize the app
start_time = time.time()


def init_app(config_path: str = None):
    """
    Initialize the app.
    
    Args:
        config_path: Path to the configuration file.
    """
    global feluda
    
    if config_path:
        feluda = Feluda(config_path)
        feluda.setup()


def run_app(host: str = "0.0.0.0", port: int = 8000, config_path: str = None):
    """
    Run the app.
    
    Args:
        host: The host to bind to.
        port: The port to bind to.
        config_path: Path to the configuration file.
    """
    init_app(config_path)
    uvicorn.run(app, host=host, port=port)
