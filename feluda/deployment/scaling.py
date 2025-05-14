"""
Scaling for Feluda.

This module provides scaling for model deployments.
"""

import abc
import enum
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.deployment.manager import DeploymentManager, get_deployment_manager
from feluda.deployment.serving import ModelServer, get_model_server
from feluda.observability import get_logger

log = get_logger(__name__)


class ScalingPolicy(BaseModel):
    """
    Scaling policy.
    
    This class represents a scaling policy for a model deployment.
    """
    
    min_replicas: int = Field(1, description="The minimum number of replicas")
    max_replicas: int = Field(10, description="The maximum number of replicas")
    target_cpu_utilization: float = Field(0.7, description="The target CPU utilization")
    target_memory_utilization: float = Field(0.7, description="The target memory utilization")
    target_requests_per_second: Optional[float] = Field(None, description="The target requests per second")
    scale_up_cooldown: float = Field(60.0, description="The scale up cooldown in seconds")
    scale_down_cooldown: float = Field(300.0, description="The scale down cooldown in seconds")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the scaling policy to a dictionary.
        
        Returns:
            A dictionary representation of the scaling policy.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalingPolicy":
        """
        Create a scaling policy from a dictionary.
        
        Args:
            data: The dictionary to create the scaling policy from.
            
        Returns:
            A scaling policy.
        """
        return cls(**data)


class AutoScaler:
    """
    Auto scaler.
    
    This class is responsible for auto-scaling model deployments.
    """
    
    def __init__(
        self,
        deployment_manager: Optional[DeploymentManager] = None,
        model_server: Optional[ModelServer] = None,
    ):
        """
        Initialize the auto scaler.
        
        Args:
            deployment_manager: The deployment manager.
            model_server: The model server.
        """
        self.deployment_manager = deployment_manager or get_deployment_manager()
        self.model_server = model_server or get_model_server()
        self.policies: Dict[str, ScalingPolicy] = {}
        self.replicas: Dict[str, int] = {}
        self.last_scale_up: Dict[str, float] = {}
        self.last_scale_down: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
    
    def set_scaling_policy(self, deployment_id: str, policy: ScalingPolicy) -> None:
        """
        Set a scaling policy for a deployment.
        
        Args:
            deployment_id: The deployment ID.
            policy: The scaling policy.
        """
        with self.lock:
            self.policies[deployment_id] = policy
            
            # Initialize the replicas
            if deployment_id not in self.replicas:
                self.replicas[deployment_id] = policy.min_replicas
            
            # Initialize the last scale timestamps
            if deployment_id not in self.last_scale_up:
                self.last_scale_up[deployment_id] = 0
            
            if deployment_id not in self.last_scale_down:
                self.last_scale_down[deployment_id] = 0
    
    def get_scaling_policy(self, deployment_id: str) -> Optional[ScalingPolicy]:
        """
        Get the scaling policy for a deployment.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            The scaling policy, or None if the deployment has no policy.
        """
        with self.lock:
            return self.policies.get(deployment_id)
    
    def get_replicas(self, deployment_id: str) -> int:
        """
        Get the number of replicas for a deployment.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            The number of replicas.
        """
        with self.lock:
            return self.replicas.get(deployment_id, 0)
    
    def scale(self, deployment_id: str, replicas: int) -> bool:
        """
        Scale a deployment.
        
        Args:
            deployment_id: The deployment ID.
            replicas: The number of replicas.
            
        Returns:
            True if the deployment was scaled, False otherwise.
        """
        with self.lock:
            # Get the deployment
            deployment = self.deployment_manager.get_deployment(deployment_id)
            
            if not deployment:
                return False
            
            # Get the scaling policy
            policy = self.get_scaling_policy(deployment_id)
            
            if not policy:
                return False
            
            # Check if the number of replicas is within the limits
            replicas = max(policy.min_replicas, min(policy.max_replicas, replicas))
            
            # Check if the number of replicas has changed
            if replicas == self.get_replicas(deployment_id):
                return True
            
            # Scale the deployment
            # In a real implementation, this would interact with the model server
            # to scale the deployment. For now, we just update the replicas.
            self.replicas[deployment_id] = replicas
            
            # Update the last scale timestamp
            now = time.time()
            
            if replicas > self.get_replicas(deployment_id):
                self.last_scale_up[deployment_id] = now
            else:
                self.last_scale_down[deployment_id] = now
            
            return True
    
    def start(self) -> None:
        """
        Start the auto scaler.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the auto scaler.
        """
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def _run(self) -> None:
        """
        Run the auto scaler.
        """
        while self.running:
            try:
                # Get all deployments
                deployments = self.deployment_manager.get_deployments()
                
                # Scale each deployment
                for deployment_id, deployment in deployments.items():
                    self._scale_deployment(deployment_id)
                
                # Sleep for a while
                time.sleep(10)
            
            except Exception as e:
                log.error(f"Error in auto scaler: {e}")
                time.sleep(10)
    
    def _scale_deployment(self, deployment_id: str) -> None:
        """
        Scale a deployment.
        
        Args:
            deployment_id: The deployment ID.
        """
        with self.lock:
            # Get the deployment
            deployment = self.deployment_manager.get_deployment(deployment_id)
            
            if not deployment:
                return
            
            # Get the scaling policy
            policy = self.get_scaling_policy(deployment_id)
            
            if not policy:
                return
            
            # Get the metrics
            metrics = self.deployment_manager.get_deployment_metrics(deployment_id)
            
            if not metrics:
                return
            
            # Calculate the desired number of replicas
            current_replicas = self.get_replicas(deployment_id)
            desired_replicas = current_replicas
            
            # Scale based on CPU utilization
            if "cpu_utilization" in metrics:
                cpu_utilization = metrics["cpu_utilization"]
                cpu_replicas = int(current_replicas * cpu_utilization / policy.target_cpu_utilization)
                desired_replicas = max(desired_replicas, cpu_replicas)
            
            # Scale based on memory utilization
            if "memory_utilization" in metrics:
                memory_utilization = metrics["memory_utilization"]
                memory_replicas = int(current_replicas * memory_utilization / policy.target_memory_utilization)
                desired_replicas = max(desired_replicas, memory_replicas)
            
            # Scale based on requests per second
            if policy.target_requests_per_second and "requests_per_second" in metrics:
                requests_per_second = metrics["requests_per_second"]
                rps_replicas = int(requests_per_second / policy.target_requests_per_second)
                desired_replicas = max(desired_replicas, rps_replicas)
            
            # Check if we can scale
            now = time.time()
            
            if desired_replicas > current_replicas:
                # Check if we can scale up
                if now - self.last_scale_up.get(deployment_id, 0) < policy.scale_up_cooldown:
                    return
            elif desired_replicas < current_replicas:
                # Check if we can scale down
                if now - self.last_scale_down.get(deployment_id, 0) < policy.scale_down_cooldown:
                    return
            
            # Scale the deployment
            self.scale(deployment_id, desired_replicas)


# Global auto scaler instance
_auto_scaler = None
_auto_scaler_lock = threading.RLock()


def get_auto_scaler() -> AutoScaler:
    """
    Get the global auto scaler instance.
    
    Returns:
        The global auto scaler instance.
    """
    global _auto_scaler
    
    with _auto_scaler_lock:
        if _auto_scaler is None:
            _auto_scaler = AutoScaler()
            _auto_scaler.start()
        
        return _auto_scaler
