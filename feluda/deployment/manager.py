"""
Deployment manager for Feluda.

This module provides a deployment manager for model serving.
"""

import abc
import enum
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.deployment.serving import ModelServer, get_model_server
from feluda.model_registry.registry import get_model_registry
from feluda.observability import get_logger

log = get_logger(__name__)


class DeploymentStatus(str, enum.Enum):
    """Enum for deployment status."""
    
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class Deployment(BaseModel):
    """
    Deployment.
    
    This class represents a model deployment.
    """
    
    id: str = Field(..., description="The deployment ID")
    name: str = Field(..., description="The deployment name")
    description: Optional[str] = Field(None, description="The deployment description")
    model_name: str = Field(..., description="The model name")
    model_version: str = Field(..., description="The model version")
    endpoint: Optional[str] = Field(None, description="The deployment endpoint")
    status: DeploymentStatus = Field(..., description="The deployment status")
    config: Dict[str, Any] = Field(default_factory=dict, description="The deployment configuration")
    metrics: Dict[str, float] = Field(default_factory=dict, description="The deployment metrics")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    deployed_at: Optional[float] = Field(None, description="The deployment timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the deployment to a dictionary.
        
        Returns:
            A dictionary representation of the deployment.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Deployment":
        """
        Create a deployment from a dictionary.
        
        Args:
            data: The dictionary to create the deployment from.
            
        Returns:
            A deployment.
        """
        return cls(**data)


class DeploymentManager:
    """
    Deployment manager.
    
    This class is responsible for managing model deployments.
    """
    
    def __init__(self, model_server: Optional[ModelServer] = None):
        """
        Initialize the deployment manager.
        
        Args:
            model_server: The model server.
        """
        self.model_server = model_server or get_model_server()
        self.model_registry = get_model_registry()
        self.deployments: Dict[str, Deployment] = {}
        self.db_path = get_config().deployment_db or "deployments/deployments.db"
        self.conn = None
        self.lock = threading.RLock()
        
        # Create the database if it doesn't exist
        self._create_database()
        
        # Load deployments from the database
        self._load_deployments()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Returns:
            A connection to the SQLite database.
        """
        if not self.conn:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        
        return self.conn
    
    def _create_database(self) -> None:
        """
        Create the SQLite database.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Create the deployments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id TEXT PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Commit the changes
            conn.commit()
    
    def _load_deployments(self) -> None:
        """
        Load deployments from the database.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get all deployments
            cursor = conn.execute("SELECT id, data FROM deployments")
            
            # Create the deployments
            for row in cursor.fetchall():
                deployment_dict = json.loads(row["data"])
                deployment = Deployment.from_dict(deployment_dict)
                self.deployments[deployment.id] = deployment
    
    def _save_deployment(self, deployment: Deployment) -> None:
        """
        Save a deployment to the database.
        
        Args:
            deployment: The deployment to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Convert the deployment to a dictionary
            deployment_dict = deployment.to_dict()
            
            # Save the deployment
            conn.execute(
                """
                INSERT OR REPLACE INTO deployments (id, data)
                VALUES (?, ?)
                """,
                (
                    deployment.id,
                    json.dumps(deployment_dict),
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def _delete_deployment(self, deployment_id: str) -> None:
        """
        Delete a deployment from the database.
        
        Args:
            deployment_id: The deployment ID.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the deployment
            conn.execute(
                "DELETE FROM deployments WHERE id = ?",
                (deployment_id,),
            )
            
            # Commit the changes
            conn.commit()
    
    def create_deployment(
        self,
        name: str,
        model_name: str,
        model_version: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Deployment:
        """
        Create a deployment.
        
        Args:
            name: The deployment name.
            model_name: The model name.
            model_version: The model version.
            description: The deployment description.
            config: The deployment configuration.
            
        Returns:
            The created deployment.
        """
        with self.lock:
            # Check if a deployment with the same name already exists
            for deployment in self.deployments.values():
                if deployment.name == name:
                    return deployment
            
            # Create the deployment
            now = time.time()
            
            deployment = Deployment(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                model_name=model_name,
                model_version=model_version,
                status=DeploymentStatus.PENDING,
                config=config or {},
                created_at=now,
                updated_at=now,
            )
            
            # Store the deployment
            self.deployments[deployment.id] = deployment
            self._save_deployment(deployment)
            
            return deployment
    
    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """
        Get a deployment by ID.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            The deployment, or None if the deployment is not found.
        """
        with self.lock:
            return self.deployments.get(deployment_id)
    
    def get_deployment_by_name(self, name: str) -> Optional[Deployment]:
        """
        Get a deployment by name.
        
        Args:
            name: The deployment name.
            
        Returns:
            The deployment, or None if the deployment is not found.
        """
        with self.lock:
            for deployment in self.deployments.values():
                if deployment.name == name:
                    return deployment
            
            return None
    
    def get_deployments(self) -> Dict[str, Deployment]:
        """
        Get all deployments.
        
        Returns:
            A dictionary mapping deployment IDs to deployments.
        """
        with self.lock:
            return self.deployments.copy()
    
    def deploy(self, deployment_id: str) -> bool:
        """
        Deploy a model.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            True if the deployment was started, False otherwise.
        """
        with self.lock:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                return False
            
            # Check if the deployment is already deployed
            if deployment.status in (DeploymentStatus.DEPLOYING, DeploymentStatus.DEPLOYED):
                return True
            
            # Update the deployment status
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.updated_at = time.time()
            self._save_deployment(deployment)
            
            # Deploy the model in a separate thread
            threading.Thread(
                target=self._deploy_model,
                args=(deployment_id,),
                daemon=True,
            ).start()
            
            return True
    
    def _deploy_model(self, deployment_id: str) -> None:
        """
        Deploy a model.
        
        Args:
            deployment_id: The deployment ID.
        """
        try:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                return
            
            # Load the model
            model = self.model_registry.load_model(deployment.model_name, deployment.model_version)
            
            if not model:
                # Update the deployment status
                with self.lock:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.updated_at = time.time()
                    self._save_deployment(deployment)
                
                return
            
            # Deploy the model
            endpoint = self.model_server.deploy_model(
                model=model,
                model_name=deployment.model_name,
                model_version=deployment.model_version,
                config=deployment.config,
            )
            
            # Update the deployment
            with self.lock:
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.endpoint = endpoint
                deployment.updated_at = time.time()
                deployment.deployed_at = time.time()
                self._save_deployment(deployment)
        
        except Exception as e:
            log.error(f"Error deploying model: {e}")
            
            # Update the deployment status
            with self.lock:
                deployment = self.get_deployment(deployment_id)
                
                if deployment:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.updated_at = time.time()
                    self._save_deployment(deployment)
    
    def undeploy(self, deployment_id: str) -> bool:
        """
        Undeploy a model.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            True if the undeployment was started, False otherwise.
        """
        with self.lock:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                return False
            
            # Check if the deployment is already undeployed
            if deployment.status in (DeploymentStatus.STOPPING, DeploymentStatus.STOPPED, DeploymentStatus.PENDING):
                return True
            
            # Update the deployment status
            deployment.status = DeploymentStatus.STOPPING
            deployment.updated_at = time.time()
            self._save_deployment(deployment)
            
            # Undeploy the model in a separate thread
            threading.Thread(
                target=self._undeploy_model,
                args=(deployment_id,),
                daemon=True,
            ).start()
            
            return True
    
    def _undeploy_model(self, deployment_id: str) -> None:
        """
        Undeploy a model.
        
        Args:
            deployment_id: The deployment ID.
        """
        try:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                return
            
            # Undeploy the model
            self.model_server.undeploy_model(
                model_name=deployment.model_name,
                model_version=deployment.model_version,
            )
            
            # Update the deployment
            with self.lock:
                deployment.status = DeploymentStatus.STOPPED
                deployment.updated_at = time.time()
                self._save_deployment(deployment)
        
        except Exception as e:
            log.error(f"Error undeploying model: {e}")
            
            # Update the deployment status
            with self.lock:
                deployment = self.get_deployment(deployment_id)
                
                if deployment:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.updated_at = time.time()
                    self._save_deployment(deployment)
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """
        Delete a deployment.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            True if the deployment was deleted, False otherwise.
        """
        with self.lock:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                return False
            
            # Check if the deployment is deployed
            if deployment.status in (DeploymentStatus.DEPLOYING, DeploymentStatus.DEPLOYED):
                # Undeploy the model
                self.undeploy(deployment_id)
            
            # Delete the deployment
            del self.deployments[deployment_id]
            self._delete_deployment(deployment_id)
            
            return True
    
    def get_deployment_metrics(self, deployment_id: str) -> Dict[str, float]:
        """
        Get deployment metrics.
        
        Args:
            deployment_id: The deployment ID.
            
        Returns:
            A dictionary mapping metric names to values.
        """
        with self.lock:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                return {}
            
            # Check if the deployment is deployed
            if deployment.status != DeploymentStatus.DEPLOYED:
                return {}
            
            # Get the metrics
            metrics = self.model_server.get_model_metrics(
                model_name=deployment.model_name,
                model_version=deployment.model_version,
            )
            
            # Update the deployment
            deployment.metrics = metrics
            deployment.updated_at = time.time()
            self._save_deployment(deployment)
            
            return metrics
    
    def predict(self, deployment_id: str, data: Any) -> Any:
        """
        Make a prediction.
        
        Args:
            deployment_id: The deployment ID.
            data: The input data.
            
        Returns:
            The prediction.
        """
        with self.lock:
            # Get the deployment
            deployment = self.get_deployment(deployment_id)
            
            if not deployment:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            # Check if the deployment is deployed
            if deployment.status != DeploymentStatus.DEPLOYED:
                raise ValueError(f"Deployment {deployment_id} is not deployed")
            
            # Make the prediction
            return self.model_server.predict(
                model_name=deployment.model_name,
                model_version=deployment.model_version,
                data=data,
            )


# Global deployment manager instance
_deployment_manager = None
_deployment_manager_lock = threading.RLock()


def get_deployment_manager() -> DeploymentManager:
    """
    Get the global deployment manager instance.
    
    Returns:
        The global deployment manager instance.
    """
    global _deployment_manager
    
    with _deployment_manager_lock:
        if _deployment_manager is None:
            _deployment_manager = DeploymentManager()
        
        return _deployment_manager
