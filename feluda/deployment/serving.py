"""
Model serving for Feluda.

This module provides model serving for machine learning models.
"""

import abc
import enum
import json
import logging
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import flask
import fastapi
import requests
import torch
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class ModelServerBackend(abc.ABC):
    """
    Base class for model server backends.
    
    This class defines the interface for model server backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def deploy_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a model.
        
        Args:
            model: The model to deploy.
            model_name: The model name.
            model_version: The model version.
            config: The deployment configuration.
            
        Returns:
            The deployment endpoint.
        """
        pass
    
    @abc.abstractmethod
    def undeploy_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Undeploy a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        pass
    
    @abc.abstractmethod
    def predict(
        self,
        model_name: str,
        model_version: str,
        data: Any,
    ) -> Any:
        """
        Make a prediction.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            data: The input data.
            
        Returns:
            The prediction.
        """
        pass
    
    @abc.abstractmethod
    def get_model_metrics(
        self,
        model_name: str,
        model_version: str,
    ) -> Dict[str, float]:
        """
        Get model metrics.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            A dictionary mapping metric names to values.
        """
        pass


class FlaskModelServer(ModelServerBackend):
    """
    Flask model server.
    
    This class implements a model server using Flask.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        """
        Initialize a Flask model server.
        
        Args:
            host: The server host.
            port: The server port.
        """
        self.host = host
        self.port = port
        self.app = flask.Flask(__name__)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.server_thread = None
        self.lock = threading.RLock()
        
        # Define the routes
        self.app.route("/predict/<model_name>/<model_version>", methods=["POST"])(self._predict_route)
        self.app.route("/metrics/<model_name>/<model_version>", methods=["GET"])(self._metrics_route)
        
        # Start the server
        self._start_server()
    
    def _start_server(self) -> None:
        """
        Start the Flask server.
        """
        if self.server_thread is None:
            self.server_thread = threading.Thread(
                target=self.app.run,
                kwargs={"host": self.host, "port": self.port},
                daemon=True,
            )
            self.server_thread.start()
    
    def _predict_route(self, model_name: str, model_version: str) -> Any:
        """
        Prediction route.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The prediction.
        """
        # Get the input data
        data = flask.request.json
        
        # Make the prediction
        try:
            result = self.predict(model_name, model_version, data)
            return flask.jsonify(result)
        except Exception as e:
            return flask.jsonify({"error": str(e)}), 500
    
    def _metrics_route(self, model_name: str, model_version: str) -> Any:
        """
        Metrics route.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The metrics.
        """
        # Get the metrics
        try:
            metrics = self.get_model_metrics(model_name, model_version)
            return flask.jsonify(metrics)
        except Exception as e:
            return flask.jsonify({"error": str(e)}), 500
    
    def _get_model_key(self, model_name: str, model_version: str) -> str:
        """
        Get the model key.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The model key.
        """
        return f"{model_name}:{model_version}"
    
    def deploy_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a model.
        
        Args:
            model: The model to deploy.
            model_name: The model name.
            model_version: The model version.
            config: The deployment configuration.
            
        Returns:
            The deployment endpoint.
        """
        with self.lock:
            # Store the model
            model_key = self._get_model_key(model_name, model_version)
            self.models[model_key] = model
            
            # Initialize the metrics
            if model_name not in self.metrics:
                self.metrics[model_name] = {}
            
            self.metrics[model_name][model_version] = {
                "requests": 0,
                "errors": 0,
                "latency": 0,
            }
            
            # Return the endpoint
            return f"http://{self.host}:{self.port}/predict/{model_name}/{model_version}"
    
    def undeploy_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Undeploy a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        with self.lock:
            # Remove the model
            model_key = self._get_model_key(model_name, model_version)
            
            if model_key in self.models:
                del self.models[model_key]
            
            # Remove the metrics
            if model_name in self.metrics and model_version in self.metrics[model_name]:
                del self.metrics[model_name][model_version]
    
    def predict(
        self,
        model_name: str,
        model_version: str,
        data: Any,
    ) -> Any:
        """
        Make a prediction.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            data: The input data.
            
        Returns:
            The prediction.
        """
        with self.lock:
            # Get the model
            model_key = self._get_model_key(model_name, model_version)
            
            if model_key not in self.models:
                raise ValueError(f"Model {model_name} version {model_version} not found")
            
            model = self.models[model_key]
            
            # Update the metrics
            if model_name in self.metrics and model_version in self.metrics[model_name]:
                self.metrics[model_name][model_version]["requests"] += 1
            
            # Make the prediction
            start_time = time.time()
            
            try:
                if isinstance(model, torch.nn.Module):
                    # Convert the input to a tensor
                    if isinstance(data, list):
                        input_tensor = torch.tensor(data)
                    elif isinstance(data, dict):
                        input_tensor = {k: torch.tensor(v) for k, v in data.items()}
                    else:
                        input_tensor = torch.tensor([data])
                    
                    # Make the prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    # Convert the output to a list
                    if isinstance(output, torch.Tensor):
                        result = output.tolist()
                    else:
                        result = output
                else:
                    # Make the prediction
                    result = model.predict(data)
                
                # Update the metrics
                if model_name in self.metrics and model_version in self.metrics[model_name]:
                    latency = time.time() - start_time
                    metrics = self.metrics[model_name][model_version]
                    metrics["latency"] = (metrics["latency"] * (metrics["requests"] - 1) + latency) / metrics["requests"]
                
                return result
            
            except Exception as e:
                # Update the metrics
                if model_name in self.metrics and model_version in self.metrics[model_name]:
                    self.metrics[model_name][model_version]["errors"] += 1
                
                raise
    
    def get_model_metrics(
        self,
        model_name: str,
        model_version: str,
    ) -> Dict[str, float]:
        """
        Get model metrics.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            A dictionary mapping metric names to values.
        """
        with self.lock:
            # Get the metrics
            if model_name not in self.metrics or model_version not in self.metrics[model_name]:
                return {}
            
            return self.metrics[model_name][model_version].copy()


class FastAPIModelServer(ModelServerBackend):
    """
    FastAPI model server.
    
    This class implements a model server using FastAPI.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Initialize a FastAPI model server.
        
        Args:
            host: The server host.
            port: The server port.
        """
        self.host = host
        self.port = port
        self.app = fastapi.FastAPI()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.server_process = None
        self.lock = threading.RLock()
        
        # Define the routes
        @self.app.post("/predict/{model_name}/{model_version}")
        async def predict_route(model_name: str, model_version: str, data: Any):
            # Make the prediction
            try:
                result = self.predict(model_name, model_version, data)
                return result
            except Exception as e:
                raise fastapi.HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics/{model_name}/{model_version}")
        async def metrics_route(model_name: str, model_version: str):
            # Get the metrics
            try:
                metrics = self.get_model_metrics(model_name, model_version)
                return metrics
            except Exception as e:
                raise fastapi.HTTPException(status_code=500, detail=str(e))
        
        # Start the server
        self._start_server()
    
    def _start_server(self) -> None:
        """
        Start the FastAPI server.
        """
        if self.server_process is None:
            # Create a temporary file for the FastAPI app
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
                f.write(f"""
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {{"message": "Hello, World!"}}

if __name__ == "__main__":
    uvicorn.run(app, host="{self.host}", port={self.port})
""".encode())
                app_path = f.name
            
            # Start the server
            self.server_process = subprocess.Popen(
                ["python", app_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
    
    def _get_model_key(self, model_name: str, model_version: str) -> str:
        """
        Get the model key.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            The model key.
        """
        return f"{model_name}:{model_version}"
    
    def deploy_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a model.
        
        Args:
            model: The model to deploy.
            model_name: The model name.
            model_version: The model version.
            config: The deployment configuration.
            
        Returns:
            The deployment endpoint.
        """
        with self.lock:
            # Store the model
            model_key = self._get_model_key(model_name, model_version)
            self.models[model_key] = model
            
            # Initialize the metrics
            if model_name not in self.metrics:
                self.metrics[model_name] = {}
            
            self.metrics[model_name][model_version] = {
                "requests": 0,
                "errors": 0,
                "latency": 0,
            }
            
            # Return the endpoint
            return f"http://{self.host}:{self.port}/predict/{model_name}/{model_version}"
    
    def undeploy_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Undeploy a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        with self.lock:
            # Remove the model
            model_key = self._get_model_key(model_name, model_version)
            
            if model_key in self.models:
                del self.models[model_key]
            
            # Remove the metrics
            if model_name in self.metrics and model_version in self.metrics[model_name]:
                del self.metrics[model_name][model_version]
    
    def predict(
        self,
        model_name: str,
        model_version: str,
        data: Any,
    ) -> Any:
        """
        Make a prediction.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            data: The input data.
            
        Returns:
            The prediction.
        """
        with self.lock:
            # Get the model
            model_key = self._get_model_key(model_name, model_version)
            
            if model_key not in self.models:
                raise ValueError(f"Model {model_name} version {model_version} not found")
            
            model = self.models[model_key]
            
            # Update the metrics
            if model_name in self.metrics and model_version in self.metrics[model_name]:
                self.metrics[model_name][model_version]["requests"] += 1
            
            # Make the prediction
            start_time = time.time()
            
            try:
                if isinstance(model, torch.nn.Module):
                    # Convert the input to a tensor
                    if isinstance(data, list):
                        input_tensor = torch.tensor(data)
                    elif isinstance(data, dict):
                        input_tensor = {k: torch.tensor(v) for k, v in data.items()}
                    else:
                        input_tensor = torch.tensor([data])
                    
                    # Make the prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    # Convert the output to a list
                    if isinstance(output, torch.Tensor):
                        result = output.tolist()
                    else:
                        result = output
                else:
                    # Make the prediction
                    result = model.predict(data)
                
                # Update the metrics
                if model_name in self.metrics and model_version in self.metrics[model_name]:
                    latency = time.time() - start_time
                    metrics = self.metrics[model_name][model_version]
                    metrics["latency"] = (metrics["latency"] * (metrics["requests"] - 1) + latency) / metrics["requests"]
                
                return result
            
            except Exception as e:
                # Update the metrics
                if model_name in self.metrics and model_version in self.metrics[model_name]:
                    self.metrics[model_name][model_version]["errors"] += 1
                
                raise
    
    def get_model_metrics(
        self,
        model_name: str,
        model_version: str,
    ) -> Dict[str, float]:
        """
        Get model metrics.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            A dictionary mapping metric names to values.
        """
        with self.lock:
            # Get the metrics
            if model_name not in self.metrics or model_version not in self.metrics[model_name]:
                return {}
            
            return self.metrics[model_name][model_version].copy()


class TorchServeModelServer(ModelServerBackend):
    """
    TorchServe model server.
    
    This class implements a model server using TorchServe.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080, management_port: int = 8081):
        """
        Initialize a TorchServe model server.
        
        Args:
            host: The server host.
            port: The server port.
            management_port: The management port.
        """
        self.host = host
        self.port = port
        self.management_port = management_port
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.lock = threading.RLock()
        
        # Start TorchServe
        self._start_torchserve()
    
    def _start_torchserve(self) -> None:
        """
        Start TorchServe.
        """
        # Check if TorchServe is already running
        try:
            response = requests.get(f"http://{self.host}:{self.management_port}/ping")
            
            if response.status_code == 200:
                return
        except:
            pass
        
        # Start TorchServe
        subprocess.Popen(
            ["torchserve", "--start", "--ncs", "--model-store", "model-store"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for TorchServe to start
        for _ in range(10):
            try:
                response = requests.get(f"http://{self.host}:{self.management_port}/ping")
                
                if response.status_code == 200:
                    return
            except:
                pass
            
            time.sleep(1)
        
        raise RuntimeError("Failed to start TorchServe")
    
    def deploy_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a model.
        
        Args:
            model: The model to deploy.
            model_name: The model name.
            model_version: The model version.
            config: The deployment configuration.
            
        Returns:
            The deployment endpoint.
        """
        with self.lock:
            # Save the model
            import tempfile
            import torch
            
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save(model, f.name)
                model_path = f.name
            
            # Create a MAR file
            subprocess.run(
                [
                    "torch-model-archiver",
                    "--model-name", model_name,
                    "--version", model_version,
                    "--model-file", model_path,
                    "--serialized-file", model_path,
                    "--export-path", "model-store",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Register the model
            response = requests.post(
                f"http://{self.host}:{self.management_port}/models",
                params={
                    "model_name": model_name,
                    "url": f"{model_name}.mar",
                    "initial_workers": 1,
                    "synchronous": "true",
                },
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to register model: {response.text}")
            
            # Initialize the metrics
            if model_name not in self.metrics:
                self.metrics[model_name] = {}
            
            self.metrics[model_name][model_version] = {
                "requests": 0,
                "errors": 0,
                "latency": 0,
            }
            
            # Return the endpoint
            return f"http://{self.host}:{self.port}/predictions/{model_name}"
    
    def undeploy_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Undeploy a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        with self.lock:
            # Unregister the model
            response = requests.delete(
                f"http://{self.host}:{self.management_port}/models/{model_name}",
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to unregister model: {response.text}")
            
            # Remove the metrics
            if model_name in self.metrics and model_version in self.metrics[model_name]:
                del self.metrics[model_name][model_version]
    
    def predict(
        self,
        model_name: str,
        model_version: str,
        data: Any,
    ) -> Any:
        """
        Make a prediction.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            data: The input data.
            
        Returns:
            The prediction.
        """
        with self.lock:
            # Update the metrics
            if model_name in self.metrics and model_version in self.metrics[model_name]:
                self.metrics[model_name][model_version]["requests"] += 1
            
            # Make the prediction
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"http://{self.host}:{self.port}/predictions/{model_name}",
                    json=data,
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to make prediction: {response.text}")
                
                # Update the metrics
                if model_name in self.metrics and model_version in self.metrics[model_name]:
                    latency = time.time() - start_time
                    metrics = self.metrics[model_name][model_version]
                    metrics["latency"] = (metrics["latency"] * (metrics["requests"] - 1) + latency) / metrics["requests"]
                
                return response.json()
            
            except Exception as e:
                # Update the metrics
                if model_name in self.metrics and model_version in self.metrics[model_name]:
                    self.metrics[model_name][model_version]["errors"] += 1
                
                raise
    
    def get_model_metrics(
        self,
        model_name: str,
        model_version: str,
    ) -> Dict[str, float]:
        """
        Get model metrics.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            A dictionary mapping metric names to values.
        """
        with self.lock:
            # Get the metrics
            if model_name not in self.metrics or model_version not in self.metrics[model_name]:
                return {}
            
            return self.metrics[model_name][model_version].copy()


class ModelServer:
    """
    Model server.
    
    This class is responsible for serving models.
    """
    
    def __init__(self, backend: Optional[ModelServerBackend] = None):
        """
        Initialize the model server.
        
        Args:
            backend: The model server backend.
        """
        config = get_config()
        
        if backend:
            self.backend = backend
        elif config.model_server_type == "fastapi":
            self.backend = FastAPIModelServer(
                host=config.model_server_host or "localhost",
                port=int(config.model_server_port or 8000),
            )
        elif config.model_server_type == "torchserve":
            self.backend = TorchServeModelServer(
                host=config.model_server_host or "localhost",
                port=int(config.model_server_port or 8080),
                management_port=int(config.model_server_management_port or 8081),
            )
        else:
            self.backend = FlaskModelServer(
                host=config.model_server_host or "localhost",
                port=int(config.model_server_port or 5000),
            )
    
    def deploy_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a model.
        
        Args:
            model: The model to deploy.
            model_name: The model name.
            model_version: The model version.
            config: The deployment configuration.
            
        Returns:
            The deployment endpoint.
        """
        return self.backend.deploy_model(model, model_name, model_version, config)
    
    def undeploy_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Undeploy a model.
        
        Args:
            model_name: The model name.
            model_version: The model version.
        """
        self.backend.undeploy_model(model_name, model_version)
    
    def predict(
        self,
        model_name: str,
        model_version: str,
        data: Any,
    ) -> Any:
        """
        Make a prediction.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            data: The input data.
            
        Returns:
            The prediction.
        """
        return self.backend.predict(model_name, model_version, data)
    
    def get_model_metrics(
        self,
        model_name: str,
        model_version: str,
    ) -> Dict[str, float]:
        """
        Get model metrics.
        
        Args:
            model_name: The model name.
            model_version: The model version.
            
        Returns:
            A dictionary mapping metric names to values.
        """
        return self.backend.get_model_metrics(model_name, model_version)


# Global model server instance
_model_server = None
_model_server_lock = threading.RLock()


def get_model_server() -> ModelServer:
    """
    Get the global model server instance.
    
    Returns:
        The global model server instance.
    """
    global _model_server
    
    with _model_server_lock:
        if _model_server is None:
            _model_server = ModelServer()
        
        return _model_server
