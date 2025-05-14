"""
Cluster management for Feluda.

This module provides cluster management for distributed computing.
"""

import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.distributed.broker import BrokerManager, Message, get_broker_manager
from feluda.distributed.worker import Worker, WorkerManager, get_worker_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class ClusterStatus(str, enum.Enum):
    """Enum for cluster status."""
    
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


class ClusterNode(BaseModel):
    """
    Cluster node.
    
    This class represents a node in a cluster.
    """
    
    id: str = Field(..., description="The node ID")
    name: str = Field(..., description="The node name")
    host: str = Field(..., description="The node host")
    port: int = Field(..., description="The node port")
    status: str = Field("online", description="The node status")
    resources: Dict[str, Any] = Field(default_factory=dict, description="The node resources")
    workers: List[str] = Field(default_factory=list, description="The node workers")
    last_heartbeat: float = Field(..., description="The last heartbeat timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary.
        
        Returns:
            A dictionary representation of the node.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterNode":
        """
        Create a node from a dictionary.
        
        Args:
            data: The dictionary to create the node from.
            
        Returns:
            A node.
        """
        return cls(**data)
    
    @classmethod
    def create(cls, name: str, host: str, port: int, resources: Dict[str, Any]) -> "ClusterNode":
        """
        Create a new node.
        
        Args:
            name: The node name.
            host: The node host.
            port: The node port.
            resources: The node resources.
            
        Returns:
            A new node.
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            host=host,
            port=port,
            resources=resources,
            last_heartbeat=time.time(),
        )


class Cluster(BaseModel):
    """
    Cluster.
    
    This class represents a cluster of nodes.
    """
    
    id: str = Field(..., description="The cluster ID")
    name: str = Field(..., description="The cluster name")
    status: ClusterStatus = Field(ClusterStatus.INITIALIZING, description="The cluster status")
    nodes: Dict[str, ClusterNode] = Field(default_factory=dict, description="The cluster nodes")
    config: Dict[str, Any] = Field(default_factory=dict, description="The cluster configuration")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cluster to a dictionary.
        
        Returns:
            A dictionary representation of the cluster.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cluster":
        """
        Create a cluster from a dictionary.
        
        Args:
            data: The dictionary to create the cluster from.
            
        Returns:
            A cluster.
        """
        return cls(**data)
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> "Cluster":
        """
        Create a new cluster.
        
        Args:
            name: The cluster name.
            config: The cluster configuration.
            
        Returns:
            A new cluster.
        """
        now = time.time()
        
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            config=config,
            created_at=now,
            updated_at=now,
        )
    
    def add_node(self, node: ClusterNode) -> None:
        """
        Add a node to the cluster.
        
        Args:
            node: The node to add.
        """
        self.nodes[node.id] = node
        self.updated_at = time.time()
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the cluster.
        
        Args:
            node_id: The node ID.
            
        Returns:
            True if the node was removed, False otherwise.
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.updated_at = time.time()
            return True
        
        return False
    
    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: The node ID.
            
        Returns:
            The node, or None if the node is not found.
        """
        return self.nodes.get(node_id)
    
    def get_nodes(self) -> List[ClusterNode]:
        """
        Get all nodes.
        
        Returns:
            A list of nodes.
        """
        return list(self.nodes.values())
    
    def update_status(self) -> None:
        """
        Update the cluster status.
        """
        if not self.nodes:
            self.status = ClusterStatus.INITIALIZING
        elif all(node.status == "online" for node in self.nodes.values()):
            self.status = ClusterStatus.RUNNING
        elif any(node.status == "offline" for node in self.nodes.values()):
            self.status = ClusterStatus.DEGRADED
        else:
            self.status = ClusterStatus.RUNNING
        
        self.updated_at = time.time()


class ClusterManager:
    """
    Cluster manager.
    
    This class is responsible for managing clusters.
    """
    
    def __init__(
        self,
        broker_manager: Optional[BrokerManager] = None,
        worker_manager: Optional[WorkerManager] = None,
    ):
        """
        Initialize the cluster manager.
        
        Args:
            broker_manager: The broker manager.
            worker_manager: The worker manager.
        """
        self.broker_manager = broker_manager or get_broker_manager()
        self.worker_manager = worker_manager or get_worker_manager()
        self.clusters: Dict[str, Cluster] = {}
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        
        # Subscribe to cluster topics
        self.broker_manager.subscribe("cluster.node.register", self._handle_node_register)
        self.broker_manager.subscribe("cluster.node.heartbeat", self._handle_node_heartbeat)
        self.broker_manager.subscribe("cluster.node.status", self._handle_node_status)
    
    def create_cluster(self, name: str, config: Dict[str, Any]) -> Cluster:
        """
        Create a cluster.
        
        Args:
            name: The cluster name.
            config: The cluster configuration.
            
        Returns:
            The created cluster.
        """
        with self.lock:
            # Create a cluster
            cluster = Cluster.create(name, config)
            
            # Store the cluster
            self.clusters[cluster.id] = cluster
            
            return cluster
    
    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """
        Get a cluster by ID.
        
        Args:
            cluster_id: The cluster ID.
            
        Returns:
            The cluster, or None if the cluster is not found.
        """
        with self.lock:
            return self.clusters.get(cluster_id)
    
    def get_clusters(self) -> List[Cluster]:
        """
        Get all clusters.
        
        Returns:
            A list of clusters.
        """
        with self.lock:
            return list(self.clusters.values())
    
    def register_node(
        self,
        cluster_id: str,
        name: str,
        host: str,
        port: int,
        resources: Dict[str, Any],
    ) -> Optional[ClusterNode]:
        """
        Register a node.
        
        Args:
            cluster_id: The cluster ID.
            name: The node name.
            host: The node host.
            port: The node port.
            resources: The node resources.
            
        Returns:
            The registered node, or None if the cluster is not found.
        """
        with self.lock:
            # Get the cluster
            cluster = self.clusters.get(cluster_id)
            
            if not cluster:
                return None
            
            # Create a node
            node = ClusterNode.create(name, host, port, resources)
            
            # Add the node to the cluster
            cluster.add_node(node)
            
            # Publish a node register message
            self.broker_manager.publish(
                topic="cluster.node.register",
                payload={
                    "cluster_id": cluster_id,
                    "node": node.to_dict(),
                },
            )
            
            # Update the cluster status
            cluster.update_status()
            
            return node
    
    def send_node_heartbeat(self, cluster_id: str, node_id: str) -> bool:
        """
        Send a heartbeat for a node.
        
        Args:
            cluster_id: The cluster ID.
            node_id: The node ID.
            
        Returns:
            True if the heartbeat was sent, False otherwise.
        """
        with self.lock:
            # Get the cluster
            cluster = self.clusters.get(cluster_id)
            
            if not cluster:
                return False
            
            # Get the node
            node = cluster.get_node(node_id)
            
            if not node:
                return False
            
            # Update the node
            node.last_heartbeat = time.time()
            
            # Publish a node heartbeat message
            self.broker_manager.publish(
                topic="cluster.node.heartbeat",
                payload={
                    "cluster_id": cluster_id,
                    "node_id": node_id,
                },
            )
            
            return True
    
    def update_node_status(self, cluster_id: str, node_id: str, status: str) -> bool:
        """
        Update the status of a node.
        
        Args:
            cluster_id: The cluster ID.
            node_id: The node ID.
            status: The node status.
            
        Returns:
            True if the node status was updated, False otherwise.
        """
        with self.lock:
            # Get the cluster
            cluster = self.clusters.get(cluster_id)
            
            if not cluster:
                return False
            
            # Get the node
            node = cluster.get_node(node_id)
            
            if not node:
                return False
            
            # Update the node
            node.status = status
            
            # Publish a node status message
            self.broker_manager.publish(
                topic="cluster.node.status",
                payload={
                    "cluster_id": cluster_id,
                    "node_id": node_id,
                    "status": status,
                },
            )
            
            # Update the cluster status
            cluster.update_status()
            
            return True
    
    def start(self) -> None:
        """
        Start the cluster manager.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the cluster manager.
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
        Run the cluster manager.
        """
        while self.running:
            try:
                # Check node heartbeats
                self._check_node_heartbeats()
                
                # Sleep for a while
                time.sleep(10)
            
            except Exception as e:
                log.error(f"Error in cluster manager: {e}")
                time.sleep(10)
    
    def _check_node_heartbeats(self) -> None:
        """
        Check node heartbeats.
        """
        with self.lock:
            now = time.time()
            
            for cluster in self.clusters.values():
                for node_id, node in list(cluster.nodes.items()):
                    # Check if the node is offline
                    if now - node.last_heartbeat > 60:
                        # Update the node status
                        node.status = "offline"
                        
                        # Publish a node status message
                        self.broker_manager.publish(
                            topic="cluster.node.status",
                            payload={
                                "cluster_id": cluster.id,
                                "node_id": node_id,
                                "status": "offline",
                            },
                        )
                
                # Update the cluster status
                cluster.update_status()
    
    def _handle_node_register(self, message: Message) -> None:
        """
        Handle a node register message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the cluster ID and node
            cluster_id = message.payload.get("cluster_id")
            node_dict = message.payload.get("node")
            
            if not cluster_id or not node_dict:
                return
            
            with self.lock:
                # Get the cluster
                cluster = self.clusters.get(cluster_id)
                
                if not cluster:
                    return
                
                # Create the node
                node = ClusterNode.from_dict(node_dict)
                
                # Add the node to the cluster
                cluster.add_node(node)
                
                # Update the cluster status
                cluster.update_status()
        
        except Exception as e:
            log.error(f"Error handling node register message: {e}")
    
    def _handle_node_heartbeat(self, message: Message) -> None:
        """
        Handle a node heartbeat message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the cluster ID and node ID
            cluster_id = message.payload.get("cluster_id")
            node_id = message.payload.get("node_id")
            
            if not cluster_id or not node_id:
                return
            
            with self.lock:
                # Get the cluster
                cluster = self.clusters.get(cluster_id)
                
                if not cluster:
                    return
                
                # Get the node
                node = cluster.get_node(node_id)
                
                if not node:
                    return
                
                # Update the node
                node.last_heartbeat = time.time()
        
        except Exception as e:
            log.error(f"Error handling node heartbeat message: {e}")
    
    def _handle_node_status(self, message: Message) -> None:
        """
        Handle a node status message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the cluster ID, node ID, and status
            cluster_id = message.payload.get("cluster_id")
            node_id = message.payload.get("node_id")
            status = message.payload.get("status")
            
            if not cluster_id or not node_id or not status:
                return
            
            with self.lock:
                # Get the cluster
                cluster = self.clusters.get(cluster_id)
                
                if not cluster:
                    return
                
                # Get the node
                node = cluster.get_node(node_id)
                
                if not node:
                    return
                
                # Update the node
                node.status = status
                
                # Update the cluster status
                cluster.update_status()
        
        except Exception as e:
            log.error(f"Error handling node status message: {e}")


# Global cluster manager instance
_cluster_manager = None
_cluster_manager_lock = threading.RLock()


def get_cluster_manager() -> ClusterManager:
    """
    Get the global cluster manager instance.
    
    Returns:
        The global cluster manager instance.
    """
    global _cluster_manager
    
    with _cluster_manager_lock:
        if _cluster_manager is None:
            _cluster_manager = ClusterManager()
            _cluster_manager.start()
        
        return _cluster_manager
