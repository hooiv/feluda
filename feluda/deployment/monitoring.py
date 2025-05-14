"""
Deployment monitoring for Feluda.

This module provides monitoring for model deployments.
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
from feluda.monitoring.alerts import AlertLevel, get_alert_manager
from feluda.monitoring.metrics import get_metric_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class DeploymentMonitor:
    """
    Deployment monitor.
    
    This class is responsible for monitoring model deployments.
    """
    
    def __init__(self, deployment_manager: Optional[DeploymentManager] = None):
        """
        Initialize the deployment monitor.
        
        Args:
            deployment_manager: The deployment manager.
        """
        self.deployment_manager = deployment_manager or get_deployment_manager()
        self.alert_manager = get_alert_manager()
        self.metric_manager = get_metric_manager()
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
        # Create metrics
        self._create_metrics()
    
    def _create_metrics(self) -> None:
        """
        Create metrics.
        """
        # Create counters
        self.metric_manager.create_counter(
            name="feluda_deployment_requests_total",
            description="Total number of requests to model deployments",
            labels=["deployment_id", "model_name", "model_version"],
        )
        
        self.metric_manager.create_counter(
            name="feluda_deployment_errors_total",
            description="Total number of errors in model deployments",
            labels=["deployment_id", "model_name", "model_version"],
        )
        
        # Create gauges
        self.metric_manager.create_gauge(
            name="feluda_deployment_latency_seconds",
            description="Latency of model deployments in seconds",
            labels=["deployment_id", "model_name", "model_version"],
        )
        
        self.metric_manager.create_gauge(
            name="feluda_deployment_cpu_utilization",
            description="CPU utilization of model deployments",
            labels=["deployment_id", "model_name", "model_version"],
        )
        
        self.metric_manager.create_gauge(
            name="feluda_deployment_memory_utilization",
            description="Memory utilization of model deployments",
            labels=["deployment_id", "model_name", "model_version"],
        )
        
        self.metric_manager.create_gauge(
            name="feluda_deployment_replicas",
            description="Number of replicas of model deployments",
            labels=["deployment_id", "model_name", "model_version"],
        )
    
    def start(self) -> None:
        """
        Start the deployment monitor.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the deployment monitor.
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
        Run the deployment monitor.
        """
        while self.running:
            try:
                # Get all deployments
                deployments = self.deployment_manager.get_deployments()
                
                # Monitor each deployment
                for deployment_id, deployment in deployments.items():
                    self._monitor_deployment(deployment_id)
                
                # Sleep for a while
                time.sleep(10)
            
            except Exception as e:
                log.error(f"Error in deployment monitor: {e}")
                time.sleep(10)
    
    def _monitor_deployment(self, deployment_id: str) -> None:
        """
        Monitor a deployment.
        
        Args:
            deployment_id: The deployment ID.
        """
        try:
            # Get the deployment
            deployment = self.deployment_manager.get_deployment(deployment_id)
            
            if not deployment:
                return
            
            # Get the metrics
            metrics = self.deployment_manager.get_deployment_metrics(deployment_id)
            
            if not metrics:
                return
            
            # Update the metrics
            labels = {
                "deployment_id": deployment_id,
                "model_name": deployment.model_name,
                "model_version": deployment.model_version,
            }
            
            # Update counters
            if "requests" in metrics:
                requests_counter = self.metric_manager.get_metric("feluda_deployment_requests_total")
                
                if requests_counter:
                    requests_counter.inc(metrics["requests"], labels)
            
            if "errors" in metrics:
                errors_counter = self.metric_manager.get_metric("feluda_deployment_errors_total")
                
                if errors_counter:
                    errors_counter.inc(metrics["errors"], labels)
            
            # Update gauges
            if "latency" in metrics:
                latency_gauge = self.metric_manager.get_metric("feluda_deployment_latency_seconds")
                
                if latency_gauge:
                    latency_gauge.set(metrics["latency"], labels)
            
            if "cpu_utilization" in metrics:
                cpu_gauge = self.metric_manager.get_metric("feluda_deployment_cpu_utilization")
                
                if cpu_gauge:
                    cpu_gauge.set(metrics["cpu_utilization"], labels)
            
            if "memory_utilization" in metrics:
                memory_gauge = self.metric_manager.get_metric("feluda_deployment_memory_utilization")
                
                if memory_gauge:
                    memory_gauge.set(metrics["memory_utilization"], labels)
            
            if "replicas" in metrics:
                replicas_gauge = self.metric_manager.get_metric("feluda_deployment_replicas")
                
                if replicas_gauge:
                    replicas_gauge.set(metrics["replicas"], labels)
            
            # Check for alerts
            self._check_alerts(deployment_id, deployment, metrics)
        
        except Exception as e:
            log.error(f"Error monitoring deployment {deployment_id}: {e}")
    
    def _check_alerts(self, deployment_id: str, deployment: Any, metrics: Dict[str, float]) -> None:
        """
        Check for alerts.
        
        Args:
            deployment_id: The deployment ID.
            deployment: The deployment.
            metrics: The deployment metrics.
        """
        # Check for high error rate
        if "requests" in metrics and "errors" in metrics and metrics["requests"] > 0:
            error_rate = metrics["errors"] / metrics["requests"]
            
            if error_rate > 0.1:
                # Create an alert
                self.alert_manager.create_alert(
                    rule_id="high_error_rate",
                    level=AlertLevel.ERROR,
                    message=f"High error rate in deployment {deployment.name}",
                    details={
                        "deployment_id": deployment_id,
                        "model_name": deployment.model_name,
                        "model_version": deployment.model_version,
                        "error_rate": error_rate,
                    },
                )
        
        # Check for high latency
        if "latency" in metrics:
            latency = metrics["latency"]
            
            if latency > 1.0:
                # Create an alert
                self.alert_manager.create_alert(
                    rule_id="high_latency",
                    level=AlertLevel.WARNING,
                    message=f"High latency in deployment {deployment.name}",
                    details={
                        "deployment_id": deployment_id,
                        "model_name": deployment.model_name,
                        "model_version": deployment.model_version,
                        "latency": latency,
                    },
                )
        
        # Check for high CPU utilization
        if "cpu_utilization" in metrics:
            cpu_utilization = metrics["cpu_utilization"]
            
            if cpu_utilization > 0.9:
                # Create an alert
                self.alert_manager.create_alert(
                    rule_id="high_cpu_utilization",
                    level=AlertLevel.WARNING,
                    message=f"High CPU utilization in deployment {deployment.name}",
                    details={
                        "deployment_id": deployment_id,
                        "model_name": deployment.model_name,
                        "model_version": deployment.model_version,
                        "cpu_utilization": cpu_utilization,
                    },
                )
        
        # Check for high memory utilization
        if "memory_utilization" in metrics:
            memory_utilization = metrics["memory_utilization"]
            
            if memory_utilization > 0.9:
                # Create an alert
                self.alert_manager.create_alert(
                    rule_id="high_memory_utilization",
                    level=AlertLevel.WARNING,
                    message=f"High memory utilization in deployment {deployment.name}",
                    details={
                        "deployment_id": deployment_id,
                        "model_name": deployment.model_name,
                        "model_version": deployment.model_version,
                        "memory_utilization": memory_utilization,
                    },
                )


# Global deployment monitor instance
_deployment_monitor = None
_deployment_monitor_lock = threading.RLock()


def get_deployment_monitor() -> DeploymentMonitor:
    """
    Get the global deployment monitor instance.
    
    Returns:
        The global deployment monitor instance.
    """
    global _deployment_monitor
    
    with _deployment_monitor_lock:
        if _deployment_monitor is None:
            _deployment_monitor = DeploymentMonitor()
            _deployment_monitor.start()
        
        return _deployment_monitor
